from Xlib import X
from Xlib.protocol import rq, structs
class ConvertSelection(rq.Request):
    _request = rq.Struct(rq.Opcode(24), rq.Pad(1), rq.RequestLength(), rq.Window('requestor'), rq.Card32('selection'), rq.Card32('target'), rq.Card32('property'), rq.Card32('time'))