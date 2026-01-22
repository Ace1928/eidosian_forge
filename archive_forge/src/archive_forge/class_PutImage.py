from Xlib import X
from Xlib.protocol import rq, structs
class PutImage(rq.Request):
    _request = rq.Struct(rq.Opcode(72), rq.Set('format', 1, (X.XYBitmap, X.XYPixmap, X.ZPixmap)), rq.RequestLength(), rq.Drawable('drawable'), rq.GC('gc'), rq.Card16('width'), rq.Card16('height'), rq.Int16('dst_x'), rq.Int16('dst_y'), rq.Card8('left_pad'), rq.Card8('depth'), rq.Pad(2), rq.String8('data'))