from Xlib import X
from Xlib.protocol import rq, structs
class KillClient(rq.Request):
    _request = rq.Struct(rq.Opcode(113), rq.Pad(1), rq.RequestLength(), rq.Resource('resource'))