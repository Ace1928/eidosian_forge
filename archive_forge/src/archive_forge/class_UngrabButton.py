from Xlib import X
from Xlib.protocol import rq, structs
class UngrabButton(rq.Request):
    _request = rq.Struct(rq.Opcode(29), rq.Card8('button'), rq.RequestLength(), rq.Window('grab_window'), rq.Card16('modifiers'), rq.Pad(2))