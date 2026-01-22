from Xlib import X
from Xlib.protocol import rq, structs
class OutputPropertyNotify(rq.Event):
    _code = None
    _fields = rq.Struct(rq.Card8('type'), rq.Card8('sub_code'), rq.Card16('sequence_number'), rq.Window('window'), rq.Card32('output'), rq.Card32('atom'), rq.Card32('timestamp'), rq.Card8('state'), rq.Pad(11))