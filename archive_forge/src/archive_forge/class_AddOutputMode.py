from Xlib import X
from Xlib.protocol import rq, structs
class AddOutputMode(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(18), rq.RequestLength(), rq.Card32('output'), rq.Card32('mode'))