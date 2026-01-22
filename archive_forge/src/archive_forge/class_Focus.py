from Xlib import X
from Xlib.protocol import rq
class Focus(rq.Event):
    _code = None
    _fields = rq.Struct(rq.Card8('type'), rq.Card8('detail'), rq.Card16('sequence_number'), rq.Window('window'), rq.Card8('mode'), rq.Pad(23))