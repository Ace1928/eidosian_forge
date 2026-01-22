from Xlib import X
from Xlib.protocol import rq, structs
class ChangePointerControl(rq.Request):
    _request = rq.Struct(rq.Opcode(105), rq.Pad(1), rq.RequestLength(), rq.Int16('accel_num'), rq.Int16('accel_denum'), rq.Int16('threshold'), rq.Bool('do_accel'), rq.Bool('do_thresh'))