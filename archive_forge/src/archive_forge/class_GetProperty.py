from Xlib import X
from Xlib.protocol import rq, structs
class GetProperty(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(20), rq.Bool('delete'), rq.RequestLength(), rq.Window('window'), rq.Card32('property'), rq.Card32('type'), rq.Card32('long_offset'), rq.Card32('long_length'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Format('value', 1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('property_type'), rq.Card32('bytes_after'), rq.LengthOf('value', 4), rq.Pad(12), rq.PropertyData('value'))