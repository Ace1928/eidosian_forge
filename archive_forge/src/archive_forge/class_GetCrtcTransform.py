from Xlib import X
from Xlib.protocol import rq, structs
class GetCrtcTransform(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(27), rq.RequestLength(), rq.Card32('crtc'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('status'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Object('pending_transform', Render_Transform), rq.Bool('has_transforms'), rq.Pad(3), rq.Object('current_transform', Render_Transform), rq.Pad(4), rq.LengthOf('pending_filter_name', 2), rq.LengthOf('pending_filter_params', 2), rq.LengthOf('current_filter_name', 2), rq.LengthOf('current_filter_params', 2), rq.String8('pending_filter_name'), rq.List('pending_filter_params', rq.Card32Obj), rq.String8('current_filter_name'), rq.List('current_filter_params', rq.Card32Obj))