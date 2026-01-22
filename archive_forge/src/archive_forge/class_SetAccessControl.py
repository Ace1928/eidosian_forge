from Xlib import X
from Xlib.protocol import rq, structs
class SetAccessControl(rq.Request):
    _request = rq.Struct(rq.Opcode(111), rq.Set('mode', 1, (X.DisableAccess, X.EnableAccess)), rq.RequestLength())