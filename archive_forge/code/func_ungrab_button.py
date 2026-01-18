from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def ungrab_button(self, button, modifiers, onerror=None):
    request.UngrabButton(display=self.display, onerror=onerror, button=button, grab_window=self.id, modifiers=modifiers)