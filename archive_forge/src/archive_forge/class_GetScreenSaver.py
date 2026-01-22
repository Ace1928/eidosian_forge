from Xlib import X
from Xlib.protocol import rq, structs
class GetScreenSaver(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(108), rq.Pad(1), rq.RequestLength())
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card16('timeout'), rq.Card16('interval'), rq.Card8('prefer_blanking'), rq.Card8('allow_exposures'), rq.Pad(18))