from Xlib import X
from Xlib.protocol import rq
class FakeInput(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(2), rq.RequestLength(), rq.Set('event_type', 1, (X.KeyPress, X.KeyRelease, X.ButtonPress, X.ButtonRelease, X.MotionNotify)), rq.Card8('detail'), rq.Pad(2), rq.Card32('time'), rq.Window('root', (X.NONE,)), rq.Pad(8), rq.Int16('x'), rq.Int16('y'), rq.Pad(8))