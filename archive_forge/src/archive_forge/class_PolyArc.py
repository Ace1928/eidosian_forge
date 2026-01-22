from Xlib import X
from Xlib.protocol import rq, structs
class PolyArc(rq.Request):
    _request = rq.Struct(rq.Opcode(68), rq.Pad(1), rq.RequestLength(), rq.Drawable('drawable'), rq.GC('gc'), rq.List('arcs', structs.Arc))