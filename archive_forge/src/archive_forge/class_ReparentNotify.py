from Xlib import X
from Xlib.protocol import rq
class ReparentNotify(rq.Event):
    _code = X.ReparentNotify
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Window('event'), rq.Window('window'), rq.Window('parent'), rq.Int16('x'), rq.Int16('y'), rq.Card8('override'), rq.Pad(11))