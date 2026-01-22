from Xlib import X
from Xlib.protocol import rq, structs
class CreateGlyphCursor(rq.Request):
    _request = rq.Struct(rq.Opcode(94), rq.Pad(1), rq.RequestLength(), rq.Cursor('cid'), rq.Font('source'), rq.Font('mask'), rq.Card16('source_char'), rq.Card16('mask_char'), rq.Card16('fore_red'), rq.Card16('fore_green'), rq.Card16('fore_blue'), rq.Card16('back_red'), rq.Card16('back_green'), rq.Card16('back_blue'))