import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import monitor as vrrp_monitor
from os_ken.services.protocols.vrrp import router as vrrp_router
class VRRPManager(app_manager.OSKenApp):

    @staticmethod
    def _instance_name(interface, vrid, is_ipv6):
        ip_version = 'ipv6' if is_ipv6 else 'ipv4'
        return 'VRRP-Router-%s-%d-%s' % (str(interface), vrid, ip_version)

    def __init__(self, *args, **kwargs):
        super(VRRPManager, self).__init__(*args, **kwargs)
        self._args = args
        self._kwargs = kwargs
        self.name = vrrp_event.VRRP_MANAGER_NAME
        self._instances = {}
        self.shutdown = hub.Queue()

    def start(self):
        t = hub.spawn(self._shutdown_loop)
        super(VRRPManager, self).start()
        return t

    @handler.set_ev_cls(vrrp_event.EventVRRPConfigRequest)
    def config_request_handler(self, ev):
        config = ev.config
        interface = ev.interface
        name = self._instance_name(interface, config.vrid, config.is_ipv6)
        if name in self._instances:
            rep = vrrp_event.EventVRRPConfigReply(None, interface, config)
            self.reply_to_request(ev, rep)
            return
        statistics = VRRPStatistics(name, config.resource_id, config.statistics_interval)
        monitor = vrrp_monitor.VRRPInterfaceMonitor.factory(interface, config, name, statistics, *self._args, **self._kwargs)
        router = vrrp_router.VRRPRouter.factory(name, monitor.name, interface, config, statistics, *self._args, **self._kwargs)
        self.register_observer(vrrp_event.EventVRRPShutdownRequest, router.name)
        router.register_observer(vrrp_event.EventVRRPStateChanged, monitor.name)
        router.register_observer(vrrp_event.EventVRRPTransmitRequest, monitor.name)
        monitor.register_observer(vrrp_event.EventVRRPReceived, router.name)
        instance = VRRPInstance(name, monitor.name, config, interface)
        self._instances[name] = instance
        monitor.start()
        router.start()
        rep = vrrp_event.EventVRRPConfigReply(instance.name, interface, config)
        self.reply_to_request(ev, rep)

    def _proxy_event(self, ev):
        name = ev.instance_name
        instance = self._instances.get(name, None)
        if not instance:
            self.logger.info('unknown vrrp router %s', name)
            return
        self.send_event(instance.name, ev)

    @handler.set_ev_cls(vrrp_event.EventVRRPShutdownRequest)
    def shutdown_request_handler(self, ev):
        self._proxy_event(ev)

    @handler.set_ev_cls(vrrp_event.EventVRRPConfigChangeRequest)
    def config_change_request_handler(self, ev):
        self._proxy_event(ev)

    @handler.set_ev_cls(vrrp_event.EventVRRPStateChanged)
    def state_change_handler(self, ev):
        instance = self._instances.get(ev.instance_name, None)
        assert instance is not None
        instance.state_changed(ev.new_state)
        if ev.old_state and ev.new_state == vrrp_event.VRRP_STATE_INITIALIZE:
            self.shutdown.put(instance)

    def _shutdown_loop(self):
        app_mgr = app_manager.AppManager.get_instance()
        while self.is_active or not self.shutdown.empty():
            instance = self.shutdown.get()
            app_mgr.uninstantiate(instance.name)
            app_mgr.uninstantiate(instance.monitor_name)
            del self._instances[instance.name]

    @handler.set_ev_cls(vrrp_event.EventVRRPListRequest)
    def list_request_handler(self, ev):
        instance_name = ev.instance_name
        if instance_name is None:
            instance_list = [vrrp_event.VRRPInstance(instance.name, instance.monitor_name, instance.config, instance.interface, instance.state) for instance in self._instances.values()]
        else:
            instance = self._instances.get(instance_name, None)
            if instance is None:
                instance_list = []
            else:
                instance_list = [vrrp_event.VRRPInstance(instance_name, instance.monitor_name, instance.config, instance.interface, instance.state)]
        vrrp_list = vrrp_event.EventVRRPListReply(instance_list)
        self.reply_to_request(ev, vrrp_list)