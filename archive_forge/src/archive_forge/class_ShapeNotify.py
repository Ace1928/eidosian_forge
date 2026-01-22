from Xlib import X
from Xlib.protocol import rq, structs
class ShapeNotify(rq.Event):
    _code = None
    _fields = rq.Struct(rq.Card8('type'), rq.Set('region', 1, (ShapeBounding, ShapeClip)), rq.Card16('sequence_number'), rq.Window('window'), rq.Int16('x'), rq.Int16('y'), rq.Card16('width'), rq.Card16('height'), rq.Card32('time'), rq.Bool('shaped'), rq.Pad(11))