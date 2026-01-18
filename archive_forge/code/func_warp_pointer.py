from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def warp_pointer(self, x, y, src_window=0, src_x=0, src_y=0, src_width=0, src_height=0, onerror=None):
    request.WarpPointer(display=self.display, onerror=onerror, src_window=src_window, dst_window=self.id, src_x=src_x, src_y=src_y, src_width=src_width, src_height=src_height, dst_x=x, dst_y=y)