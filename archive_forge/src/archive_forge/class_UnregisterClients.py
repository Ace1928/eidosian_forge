from Xlib import X
from Xlib.protocol import rq
class UnregisterClients(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(3), rq.RequestLength(), rq.Card32('context'), rq.LengthOf('clients', 4), rq.List('clients', rq.Card32Obj))