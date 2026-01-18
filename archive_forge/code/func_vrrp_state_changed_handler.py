from os_ken import cfg
import socket
import netaddr
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
from os_ken.lib import rpc
from os_ken.lib import hub
from os_ken.lib import mac
@handler.set_ev_cls(vrrp_event.EventVRRPStateChanged)
def vrrp_state_changed_handler(self, ev):
    self.logger.info('handle EventVRRPStateChanged')
    name = ev.instance_name
    old_state = ev.old_state
    new_state = ev.new_state
    vrid = ev.config.vrid
    self.logger.info('VRID:%s %s: %s -> %s', vrid, name, old_state, new_state)
    params = {'vrid': vrid, 'old_state': old_state, 'new_state': new_state}
    for peer in self._peers:
        peer._endpoint.send_notification('notify_status', [params])