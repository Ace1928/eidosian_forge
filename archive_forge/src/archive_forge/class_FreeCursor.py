from Xlib import X
from Xlib.protocol import rq, structs
class FreeCursor(rq.Request):
    _request = rq.Struct(rq.Opcode(95), rq.Pad(1), rq.RequestLength(), rq.Cursor('cursor'))