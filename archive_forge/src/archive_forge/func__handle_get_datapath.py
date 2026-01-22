import numbers
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER,\
from os_ken.controller.handler import set_ev_cls
from . import event
from . import exception
@set_ev_cls(event.GetDatapathRequest, MAIN_DISPATCHER)
def _handle_get_datapath(self, req):
    result = None
    if req.dpid is None:
        result = [v.datapath for v in self._switches.values()]
    elif req.dpid in self._switches:
        result = self._switches[req.dpid].datapath
    self.reply_to_request(req, event.Reply(result=result))