from Xlib import X
from Xlib.protocol import rq, structs
class GetGeometry(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(14), rq.Pad(1), rq.RequestLength(), rq.Drawable('drawable'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('depth'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Window('root'), rq.Int16('x'), rq.Int16('y'), rq.Card16('width'), rq.Card16('height'), rq.Card16('border_width'), rq.Pad(10))