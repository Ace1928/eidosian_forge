from Xlib import X
from Xlib.protocol import rq, structs
class ForceScreenSaver(rq.Request):
    _request = rq.Struct(rq.Opcode(115), rq.Set('mode', 1, (X.ScreenSaverReset, X.ScreenSaverActive)), rq.RequestLength())