import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
class VRRPRouter(app_manager.OSKenApp):
    _EVENTS = [vrrp_event.EventVRRPStateChanged]
    _CONSTRUCTORS = {}
    _STATE_MAP = {}

    @staticmethod
    def register(version):

        def _register(cls):
            VRRPRouter._CONSTRUCTORS[version] = cls
            return cls
        return _register

    @staticmethod
    def factory(name, monitor_name, interface, config, statistics, *args, **kwargs):
        cls = VRRPRouter._CONSTRUCTORS[config.version]
        app_mgr = app_manager.AppManager.get_instance()
        kwargs = kwargs.copy()
        kwargs['name'] = name
        kwargs['monitor_name'] = monitor_name
        kwargs['vrrp_interface'] = interface
        kwargs['vrrp_config'] = config
        kwargs['vrrp_statistics'] = statistics
        return app_mgr.instantiate(cls, *args, **kwargs)

    class _EventMasterDown(event.EventBase):
        pass

    class _EventAdver(event.EventBase):
        pass

    class _EventPreemptDelay(event.EventBase):
        pass

    class _EventStatisticsOut(event.EventBase):
        pass

    def __init__(self, *args, **kwargs):
        super(VRRPRouter, self).__init__(*args, **kwargs)
        self.name = kwargs['name']
        self.monitor_name = kwargs['monitor_name']
        self.interface = kwargs['vrrp_interface']
        self.config = kwargs['vrrp_config']
        self.statistics = kwargs['vrrp_statistics']
        self.params = VRRPParams(self.config)
        self.state = None
        self.state_impl = None
        self.vrrp = None
        self.master_down_timer = TimerEventSender(self, self._EventMasterDown)
        self.adver_timer = TimerEventSender(self, self._EventAdver)
        self.preempt_delay_timer = TimerEventSender(self, self._EventPreemptDelay)
        self.register_observer(self._EventMasterDown, self.name)
        self.register_observer(self._EventAdver, self.name)
        self.stats_out_timer = TimerEventSender(self, self._EventStatisticsOut)
        self.register_observer(self._EventStatisticsOut, self.name)

    def send_advertisement(self, release=False):
        if self.vrrp is None:
            config = self.config
            max_adver_int = vrrp.vrrp.sec_to_max_adver_int(config.version, config.advertisement_interval)
            self.vrrp = vrrp.vrrp.create_version(config.version, vrrp.VRRP_TYPE_ADVERTISEMENT, config.vrid, config.priority, max_adver_int, config.ip_addresses)
        vrrp_ = self.vrrp
        if release:
            vrrp_ = vrrp_.create(vrrp_.type, vrrp_.vrid, vrrp.VRRP_PRIORITY_RELEASE_RESPONSIBILITY, vrrp_.max_adver_int, vrrp_.ip_addresses)
        if self.vrrp.priority == 0:
            self.statistics.tx_vrrp_zero_prio_packets += 1
        interface = self.interface
        packet_ = vrrp_.create_packet(interface.primary_ip_address, interface.vlan_id)
        packet_.serialize()
        vrrp_api.vrrp_transmit(self, self.monitor_name, packet_.data)
        self.statistics.tx_vrrp_packets += 1

    def state_change(self, new_state):
        old_state = self.state
        self.state = new_state
        self.state_impl = self._STATE_MAP[new_state](self)
        state_changed = vrrp_event.EventVRRPStateChanged(self.name, self.monitor_name, self.interface, self.config, old_state, new_state)
        self.send_event_to_observers(state_changed)

    @handler.set_ev_handler(_EventMasterDown)
    def master_down_handler(self, ev):
        self.state_impl.master_down(ev)

    @handler.set_ev_handler(_EventAdver)
    def adver_handler(self, ev):
        self.state_impl.adver(ev)

    @handler.set_ev_handler(_EventPreemptDelay)
    def preempt_delay_handler(self, ev):
        self.state_impl.preempt_delay(ev)

    @handler.set_ev_handler(vrrp_event.EventVRRPReceived)
    def vrrp_received_handler(self, ev):
        self.state_impl.vrrp_received(ev)

    @handler.set_ev_handler(vrrp_event.EventVRRPShutdownRequest)
    def vrrp_shutdown_request_handler(self, ev):
        assert ev.instance_name == self.name
        self.state_impl.vrrp_shutdown_request(ev)

    @handler.set_ev_handler(vrrp_event.EventVRRPConfigChangeRequest)
    def vrrp_config_change_request_handler(self, ev):
        config = self.config
        if ev.priority is not None:
            config.priority = ev.priority
        if ev.advertisement_interval is not None:
            config.advertisement_interval = ev.advertisement_interval
        if ev.preempt_mode is not None:
            config.preempt_mode = ev.preempt_mode
        if ev.preempt_delay is not None:
            config.preempt_delay = ev.preempt_delay
        if ev.accept_mode is not None:
            config.accept_mode = ev.accept_mode
        self.vrrp = None
        self.state_impl.vrrp_config_change_request(ev)

    @handler.set_ev_handler(_EventStatisticsOut)
    def statistics_handler(self, ev):
        self.stats_out_timer.start(self.statistics.statistics_interval)