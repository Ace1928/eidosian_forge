from Xlib import X
from Xlib.protocol import rq
class GetVersion(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(0), rq.RequestLength(), rq.Card8('major_version'), rq.Pad(1), rq.Card16('minor_version'))
    _reply = rq.Struct(rq.Pad(1), rq.Card8('major_version'), rq.Card16('sequence_number'), rq.Pad(4), rq.Card16('minor_version'), rq.Pad(22))