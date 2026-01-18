import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def ungrab_server(self, onerror=None):
    """Release the server if it was previously grabbed by this client."""
    request.UngrabServer(display=self.display, onerror=onerror)