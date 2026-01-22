from Xlib import X
from Xlib.protocol import rq, structs
class RotateProperties(rq.Request):
    _request = rq.Struct(rq.Opcode(114), rq.Pad(1), rq.RequestLength(), rq.Window('window'), rq.LengthOf('properties', 2), rq.Int16('delta'), rq.List('properties', rq.Card32Obj))