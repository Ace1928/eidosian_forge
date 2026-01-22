from Xlib import X
from Xlib.protocol import rq, structs
class PolyText8(rq.Request):
    _request = rq.Struct(rq.Opcode(74), rq.Pad(1), rq.RequestLength(), rq.Drawable('drawable'), rq.GC('gc'), rq.Int16('x'), rq.Int16('y'), rq.TextElements8('items'))