from Xlib import X
from Xlib.protocol import rq, structs
class FreePixmap(rq.Request):
    _request = rq.Struct(rq.Opcode(54), rq.Pad(1), rq.RequestLength(), rq.Pixmap('pixmap'))