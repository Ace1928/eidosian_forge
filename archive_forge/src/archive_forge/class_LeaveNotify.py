from Xlib import X
from Xlib.protocol import rq
class LeaveNotify(EnterLeave):
    _code = X.LeaveNotify