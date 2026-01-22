import inspect
import time
from os_ken.controller import handler
from os_ken import ofproto
from . import event
def _ofp_msg_name_to_ev_name(msg_name):
    return 'Event' + msg_name