from Xlib import X
from Xlib.protocol import rq, structs
class GetAtomName(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(17), rq.Pad(1), rq.RequestLength(), rq.Card32('atom'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.LengthOf('name', 2), rq.Pad(22), rq.String8('name'))