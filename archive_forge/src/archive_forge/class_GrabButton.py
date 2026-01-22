from Xlib import X
from Xlib.protocol import rq, structs
class GrabButton(rq.Request):
    _request = rq.Struct(rq.Opcode(28), rq.Bool('owner_events'), rq.RequestLength(), rq.Window('grab_window'), rq.Card16('event_mask'), rq.Set('pointer_mode', 1, (X.GrabModeSync, X.GrabModeAsync)), rq.Set('keyboard_mode', 1, (X.GrabModeSync, X.GrabModeAsync)), rq.Window('confine_to', (X.NONE,)), rq.Cursor('cursor', (X.NONE,)), rq.Card8('button'), rq.Pad(1), rq.Card16('modifiers'))