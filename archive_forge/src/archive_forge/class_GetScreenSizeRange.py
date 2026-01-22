from Xlib import X
from Xlib.protocol import rq, structs
class GetScreenSizeRange(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(6), rq.RequestLength(), rq.Window('window'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card16('min_width'), rq.Card16('min_height'), rq.Card16('max_width'), rq.Card16('max_height'), rq.Pad(16))