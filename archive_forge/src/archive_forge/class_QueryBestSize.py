from Xlib import X
from Xlib.protocol import rq, structs
class QueryBestSize(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(97), rq.Set('item_class', 1, (X.CursorShape, X.TileShape, X.StippleShape)), rq.RequestLength(), rq.Drawable('drawable'), rq.Card16('width'), rq.Card16('height'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card16('width'), rq.Card16('height'), rq.Pad(20))