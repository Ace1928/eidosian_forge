from Xlib import X
from Xlib.protocol import rq, structs
class ScreenChangeNotify(rq.Event):
    _code = None
    _fields = rq.Struct(rq.Card8('type'), rq.Card8('rotation'), rq.Card16('sequence_number'), rq.Card32('timestamp'), rq.Card32('config_timestamp'), rq.Window('root'), rq.Window('window'), rq.Card16('size_id'), rq.Card16('subpixel_order'), rq.Card16('width_in_pixels'), rq.Card16('height_in_pixels'), rq.Card16('width_in_millimeters'), rq.Card16('height_in_millimeters'))