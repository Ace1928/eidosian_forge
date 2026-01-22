from Xlib import X
from Xlib.protocol import rq, structs
class GetCrtcGamma(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(23), rq.RequestLength(), rq.Card32('crtc'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('status'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card16('size'), rq.Pad(22), rq.List('red', rq.Card16Obj), rq.List('green', rq.Card16Obj), rq.List('blue', rq.Card16Obj))