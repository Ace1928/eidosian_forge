from Xlib import X
from Xlib.protocol import rq
class SelectionNotify(rq.Event):
    _code = X.SelectionNotify
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Card32('time'), rq.Window('requestor'), rq.Card32('selection'), rq.Card32('target'), rq.Card32('property'), rq.Pad(8))