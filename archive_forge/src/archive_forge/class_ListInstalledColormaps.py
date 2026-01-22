from Xlib import X
from Xlib.protocol import rq, structs
class ListInstalledColormaps(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(83), rq.Pad(1), rq.RequestLength(), rq.Window('window'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.LengthOf('cmaps', 2), rq.Pad(22), rq.List('cmaps', rq.ColormapObj))