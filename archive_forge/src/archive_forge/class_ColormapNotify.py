from Xlib import X
from Xlib.protocol import rq
class ColormapNotify(rq.Event):
    _code = X.ColormapNotify
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Window('window'), rq.Colormap('colormap', (X.NONE,)), rq.Card8('new'), rq.Card8('state'), rq.Pad(18))