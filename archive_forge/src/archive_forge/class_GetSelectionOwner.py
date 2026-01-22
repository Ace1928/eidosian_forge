from Xlib import X
from Xlib.protocol import rq, structs
class GetSelectionOwner(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(23), rq.Pad(1), rq.RequestLength(), rq.Card32('selection'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Window('owner', (X.NONE,)), rq.Pad(20))