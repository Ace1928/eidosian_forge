from Xlib import X
from Xlib.protocol import rq, structs
class MapSubwindows(rq.Request):
    _request = rq.Struct(rq.Opcode(9), rq.Pad(1), rq.RequestLength(), rq.Window('window'))