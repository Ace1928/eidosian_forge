from Xlib import X
from Xlib.protocol import rq, structs
class GrabPointer(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(26), rq.Bool('owner_events'), rq.RequestLength(), rq.Window('grab_window'), rq.Card16('event_mask'), rq.Set('pointer_mode', 1, (X.GrabModeSync, X.GrabModeAsync)), rq.Set('keyboard_mode', 1, (X.GrabModeSync, X.GrabModeAsync)), rq.Window('confine_to', (X.NONE,)), rq.Cursor('cursor', (X.NONE,)), rq.Card32('time'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('status'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Pad(24))