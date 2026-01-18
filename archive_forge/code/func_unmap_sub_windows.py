from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def unmap_sub_windows(self, onerror=None):
    request.UnmapSubwindows(display=self.display, onerror=onerror, window=self.id)