from Xlib import X
from Xlib.protocol import rq, structs
class ListProperties(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(21), rq.Pad(1), rq.RequestLength(), rq.Window('window'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.LengthOf('atoms', 2), rq.Pad(22), rq.List('atoms', rq.Card32Obj))