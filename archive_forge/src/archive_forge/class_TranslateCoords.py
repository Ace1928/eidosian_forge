from Xlib import X
from Xlib.protocol import rq, structs
class TranslateCoords(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(40), rq.Pad(1), rq.RequestLength(), rq.Window('src_wid'), rq.Window('dst_wid'), rq.Int16('src_x'), rq.Int16('src_y'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('same_screen'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Window('child', (X.NONE,)), rq.Int16('x'), rq.Int16('y'), rq.Pad(16))