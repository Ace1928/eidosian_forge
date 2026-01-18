import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
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