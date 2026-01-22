from Xlib import X
from Xlib.protocol import rq, structs
class CreateMode(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(16), rq.RequestLength(), rq.Window('window'), rq.Object('mode', RandR_ModeInfo), rq.String8('name'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('mode'), rq.Pad(20))