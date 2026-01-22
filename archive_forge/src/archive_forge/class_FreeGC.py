from Xlib import X
from Xlib.protocol import rq, structs
class FreeGC(rq.Request):
    _request = rq.Struct(rq.Opcode(60), rq.Pad(1), rq.RequestLength(), rq.GC('gc'))