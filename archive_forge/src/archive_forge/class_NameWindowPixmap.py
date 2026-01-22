from Xlib import X
from Xlib.protocol import rq
from Xlib.xobject import drawable
class NameWindowPixmap(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(6), rq.RequestLength(), rq.Window('window'), rq.Pixmap('pixmap'))