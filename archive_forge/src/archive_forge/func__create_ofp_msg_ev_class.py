import inspect
import time
from os_ken.controller import handler
from os_ken import ofproto
from . import event
def _create_ofp_msg_ev_class(msg_cls):
    name = _ofp_msg_name_to_ev_name(msg_cls.__name__)
    if name in _OFP_MSG_EVENTS:
        return
    cls = type(name, (EventOFPMsgBase,), dict(__init__=lambda self, msg: super(self.__class__, self).__init__(msg)))
    globals()[name] = cls
    _OFP_MSG_EVENTS[name] = cls