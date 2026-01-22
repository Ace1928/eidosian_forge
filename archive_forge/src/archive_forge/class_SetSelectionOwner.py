from Xlib import X
from Xlib.protocol import rq, structs
class SetSelectionOwner(rq.Request):
    _request = rq.Struct(rq.Opcode(22), rq.Pad(1), rq.RequestLength(), rq.Window('window'), rq.Card32('selection'), rq.Card32('time'))