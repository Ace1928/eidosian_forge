from Xlib import X
from Xlib.protocol import rq, structs
class GetOutputProperty(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(15), rq.RequestLength(), rq.Card32('output'), rq.Card32('property'), rq.Card32('type'), rq.Card32('long_offset'), rq.Card32('long_length'), rq.Bool('delete'), rq.Bool('pending'), rq.Pad(2))
    _reply = rq.Struct(rq.ReplyCode(), rq.Format('value', 1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('property_type'), rq.Card32('bytes_after'), rq.LengthOf('value', 4), rq.Pad(12), rq.List('value', rq.Card8Obj))