from Xlib import X
from Xlib.protocol import rq, structs
class ClearArea(rq.Request):
    _request = rq.Struct(rq.Opcode(61), rq.Bool('exposures'), rq.RequestLength(), rq.Window('window'), rq.Int16('x'), rq.Int16('y'), rq.Card16('width'), rq.Card16('height'))