from Xlib import X
from Xlib.protocol import rq
class MapNotify(rq.Event):
    _code = X.MapNotify
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Window('event'), rq.Window('window'), rq.Card8('override'), rq.Pad(19))