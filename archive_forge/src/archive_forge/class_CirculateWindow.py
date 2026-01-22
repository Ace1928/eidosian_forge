from Xlib import X
from Xlib.protocol import rq, structs
class CirculateWindow(rq.Request):
    _request = rq.Struct(rq.Opcode(13), rq.Set('direction', 1, (X.RaiseLowest, X.LowerHighest)), rq.RequestLength(), rq.Window('window'))