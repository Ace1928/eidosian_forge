from Xlib import X
from Xlib.protocol import rq
def WindowValues(arg):
    return rq.ValueList(arg, 4, 0, rq.Pixmap('background_pixmap'), rq.Card32('background_pixel'), rq.Pixmap('border_pixmap'), rq.Card32('border_pixel'), rq.Gravity('bit_gravity'), rq.Gravity('win_gravity'), rq.Set('backing_store', 1, (X.NotUseful, X.WhenMapped, X.Always)), rq.Card32('backing_planes'), rq.Card32('backing_pixel'), rq.Bool('override_redirect'), rq.Bool('save_under'), rq.Card32('event_mask'), rq.Card32('do_not_propagate_mask'), rq.Colormap('colormap'), rq.Cursor('cursor'))