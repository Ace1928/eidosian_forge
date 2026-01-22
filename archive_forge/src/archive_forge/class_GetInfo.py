from Xlib import X
from Xlib.protocol import rq, structs
class GetInfo(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(4), rq.RequestLength(), rq.Card32('visual'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Window('window'))