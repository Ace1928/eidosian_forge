from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import sample_router
def _router_factory(self, instance_name, monitor_name, interface, config):
    cls = None
    for interface_cls, router_clses in self._ROUTER_CLASSES.items():
        if isinstance(interface, interface_cls):
            if config.is_ipv6:
                cls = router_clses[6]
            else:
                cls = router_clses[4]
            break
    self.logger.debug('interface %s %s', type(interface), interface)
    self.logger.debug('cls %s', cls)
    if cls is None:
        raise ValueError('Unknown interface type %s %s' % (type(interface), interface))
    kwargs = self._kwargs.copy()
    kwargs.update({'name': instance_name, 'monitor_name': monitor_name, 'config': config, 'interface': interface})
    app_mgr = app_manager.AppManager.get_instance()
    return app_mgr.instantiate(cls, *self._args, **kwargs)