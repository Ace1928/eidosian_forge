from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import dpid as lib_dpid
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import api as vrrp_api
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import monitor_openflow
from os_ken.topology import event as topo_event
from os_ken.topology import api as topo_api
from . import vrrp_common
@handler.set_ev_cls(topo_event.EventSwitchEnter)
def _switch_enter_handler(self, ev):
    if self.start_main:
        return
    switches = topo_api.get_switch(self)
    if len(switches) < 2:
        return
    self.start_main = True
    app_mgr = app_manager.AppManager.get_instance()
    self.logger.debug('%s', app_mgr.applications)
    self.switches = app_mgr.applications['switches']
    hub.spawn(self._main)