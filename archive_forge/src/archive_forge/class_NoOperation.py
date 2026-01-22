from Xlib import X
from Xlib.protocol import rq, structs
class NoOperation(rq.Request):
    _request = rq.Struct(rq.Opcode(127), rq.Pad(1), rq.RequestLength())