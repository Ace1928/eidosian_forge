from Xlib import X
from Xlib.protocol import rq, structs
class _1_0SetScreenConfig(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(2), rq.RequestLength(), rq.Drawable('drawable'), rq.Card32('timestamp'), rq.Card32('config_timestamp'), rq.Card16('size_id'), rq.Card16('rotation'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('status'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('new_timestamp'), rq.Card32('new_config_timestamp'), rq.Window('root'), rq.Card16('subpixel_order'), rq.Pad(10))