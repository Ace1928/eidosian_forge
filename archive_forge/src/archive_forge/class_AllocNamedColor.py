from Xlib import X
from Xlib.protocol import rq, structs
class AllocNamedColor(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(85), rq.Pad(1), rq.RequestLength(), rq.Colormap('cmap'), rq.LengthOf('name', 2), rq.Pad(2), rq.String8('name'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('pixel'), rq.Card16('exact_red'), rq.Card16('exact_green'), rq.Card16('exact_blue'), rq.Card16('screen_red'), rq.Card16('screen_green'), rq.Card16('screen_blue'), rq.Pad(8))