from Xlib import X
from Xlib.protocol import rq
class FreeContext(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(7), rq.RequestLength(), rq.Card32('context'))