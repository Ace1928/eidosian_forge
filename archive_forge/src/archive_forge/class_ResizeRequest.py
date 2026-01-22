from Xlib import X
from Xlib.protocol import rq
class ResizeRequest(rq.Event):
    _code = X.ResizeRequest
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Window('window'), rq.Card16('width'), rq.Card16('height'), rq.Pad(20))