from Xlib import X
from Xlib.protocol import rq, structs
class GrabServer(rq.Request):
    _request = rq.Struct(rq.Opcode(36), rq.Pad(1), rq.RequestLength())