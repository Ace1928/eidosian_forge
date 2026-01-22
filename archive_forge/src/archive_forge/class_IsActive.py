from Xlib import X
from Xlib.protocol import rq, structs
class IsActive(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(4), rq.RequestLength())
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('state'), rq.Pad(20))