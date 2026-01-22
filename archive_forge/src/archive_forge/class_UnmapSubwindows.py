from Xlib import X
from Xlib.protocol import rq, structs
class UnmapSubwindows(rq.Request):
    _request = rq.Struct(rq.Opcode(11), rq.Pad(1), rq.RequestLength(), rq.Window('window'))