from Xlib import X
from Xlib.protocol import rq, structs
class ChangeWindowAttributes(rq.Request):
    _request = rq.Struct(rq.Opcode(2), rq.Pad(1), rq.RequestLength(), rq.Window('window'), structs.WindowValues('attrs'))