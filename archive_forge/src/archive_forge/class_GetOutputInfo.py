from Xlib import X
from Xlib.protocol import rq, structs
class GetOutputInfo(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(9), rq.RequestLength(), rq.Card32('output'), rq.Card32('config_timestamp'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('status'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('timestamp'), rq.Card32('crtc'), rq.Card32('mm_width'), rq.Card32('mm_height'), rq.Card8('connection'), rq.Card8('subpixel_order'), rq.LengthOf('crtcs', 2), rq.LengthOf('modes', 2), rq.LengthOf('preferred', 2), rq.LengthOf('clones', 2), rq.LengthOf('name', 2), rq.List('crtcs', rq.Card32Obj), rq.List('modes', rq.Card32Obj), rq.List('preferred', rq.Card32Obj), rq.List('clones', rq.Card32Obj), rq.String8('name'))