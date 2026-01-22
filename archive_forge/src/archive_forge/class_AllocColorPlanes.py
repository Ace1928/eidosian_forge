from Xlib import X
from Xlib.protocol import rq, structs
class AllocColorPlanes(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(87), rq.Bool('contiguous'), rq.RequestLength(), rq.Colormap('cmap'), rq.Card16('colors'), rq.Card16('red'), rq.Card16('green'), rq.Card16('blue'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.LengthOf('pixels', 2), rq.Pad(2), rq.Card32('red_mask'), rq.Card32('green_mask'), rq.Card32('blue_mask'), rq.Pad(8), rq.List('pixels', rq.Card32Obj))