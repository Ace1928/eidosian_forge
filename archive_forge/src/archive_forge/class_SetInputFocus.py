from Xlib import X
from Xlib.protocol import rq, structs
class SetInputFocus(rq.Request):
    _request = rq.Struct(rq.Opcode(42), rq.Set('revert_to', 1, (X.RevertToNone, X.RevertToPointerRoot, X.RevertToParent)), rq.RequestLength(), rq.Window('focus'), rq.Card32('time'))