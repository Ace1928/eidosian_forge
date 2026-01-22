from Xlib import X
from Xlib.protocol import rq, structs
class UngrabPointer(rq.Request):
    _request = rq.Struct(rq.Opcode(27), rq.Pad(1), rq.RequestLength(), rq.Card32('time'))