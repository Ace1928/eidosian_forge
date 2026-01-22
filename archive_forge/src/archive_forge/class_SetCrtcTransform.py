from Xlib import X
from Xlib.protocol import rq, structs
class SetCrtcTransform(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(26), rq.RequestLength(), rq.Card32('crtc'), rq.Object('transform', Render_Transform), rq.LengthOf('filter_name', 2), rq.Pad(2), rq.String8('filter_name'), rq.List('filter_params', rq.Card32Obj))