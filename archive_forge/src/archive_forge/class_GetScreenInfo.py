from Xlib import X
from Xlib.protocol import rq, structs
class GetScreenInfo(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(5), rq.RequestLength(), rq.Window('window'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('set_of_rotations'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Window('root'), rq.Card32('timestamp'), rq.Card32('config_timestamp'), rq.LengthOf('sizes', 2), rq.Card16('size_id'), rq.Card16('rotation'), rq.Card16('rate'), rq.Card16('n_rate_ents'), rq.Pad(2), rq.List('sizes', RandR_ScreenSizes))