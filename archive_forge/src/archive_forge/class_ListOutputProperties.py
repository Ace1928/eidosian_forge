from Xlib import X
from Xlib.protocol import rq, structs
class ListOutputProperties(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(10), rq.RequestLength(), rq.Card32('output'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.LengthOf('atoms', 2), rq.Pad(22), rq.List('atoms', rq.Card32Obj))