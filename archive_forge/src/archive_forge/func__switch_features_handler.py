import numbers
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER,\
from os_ken.controller.handler import set_ev_cls
from . import event
from . import exception
@set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
def _switch_features_handler(self, ev):
    datapath = ev.msg.datapath
    id = datapath.id
    assert isinstance(id, numbers.Integral)
    old_info = self._switches.get(id, None)
    new_info = _SwitchInfo(datapath=datapath)
    self.logger.debug('add dpid %s datapath %s new_info %s old_info %s', id, datapath, new_info, old_info)
    self._switches[id] = new_info
    if old_info:
        old_info.datapath.close()
        for xid in list(old_info.barriers):
            self._cancel(old_info, xid, exception.InvalidDatapath(result=id))