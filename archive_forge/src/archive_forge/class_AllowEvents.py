from Xlib import X
from Xlib.protocol import rq, structs
class AllowEvents(rq.Request):
    _request = rq.Struct(rq.Opcode(35), rq.Set('mode', 1, (X.AsyncPointer, X.SyncPointer, X.ReplayPointer, X.AsyncKeyboard, X.SyncKeyboard, X.ReplayKeyboard, X.AsyncBoth, X.SyncBoth)), rq.RequestLength(), rq.Card32('time'))