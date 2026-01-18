import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def ungrab_pointer(self, time, onerror=None):
    """elease a grabbed pointer and any queued events. See
        XUngrabPointer(3X11)."""
    request.UngrabPointer(display=self.display, onerror=onerror, time=time)