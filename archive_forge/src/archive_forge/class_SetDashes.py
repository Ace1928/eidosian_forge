from Xlib import X
from Xlib.protocol import rq, structs
class SetDashes(rq.Request):
    _request = rq.Struct(rq.Opcode(58), rq.Pad(1), rq.RequestLength(), rq.GC('gc'), rq.Card16('dash_offset'), rq.LengthOf('dashes', 2), rq.List('dashes', rq.Card8Obj))