from Xlib import X
from Xlib.protocol import rq, structs
class QueryExtension(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(98), rq.Pad(1), rq.RequestLength(), rq.LengthOf('name', 2), rq.Pad(2), rq.String8('name'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card8('present'), rq.Card8('major_opcode'), rq.Card8('first_event'), rq.Card8('first_error'), rq.Pad(20))