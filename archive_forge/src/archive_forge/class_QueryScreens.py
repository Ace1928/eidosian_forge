from Xlib import X
from Xlib.protocol import rq, structs
class QueryScreens(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(5), rq.RequestLength())
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('number'), rq.Pad(20), rq.List('screens', structs.Rectangle))