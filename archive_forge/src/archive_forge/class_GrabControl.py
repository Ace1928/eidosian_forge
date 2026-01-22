from Xlib import X
from Xlib.protocol import rq
class GrabControl(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(3), rq.RequestLength(), rq.Bool('impervious'), rq.Pad(3))