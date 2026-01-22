from Xlib import X
from Xlib.protocol import rq, structs
class RecolorCursor(rq.Request):
    _request = rq.Struct(rq.Opcode(96), rq.Pad(1), rq.RequestLength(), rq.Cursor('cursor'), rq.Card16('fore_red'), rq.Card16('fore_green'), rq.Card16('fore_blue'), rq.Card16('back_red'), rq.Card16('back_green'), rq.Card16('back_blue'))