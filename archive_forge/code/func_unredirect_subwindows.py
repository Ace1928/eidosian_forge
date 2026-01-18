from Xlib import X
from Xlib.protocol import rq
from Xlib.xobject import drawable
def unredirect_subwindows(self, update):
    """Stop redirecting the hierarchies of children to this window.
    """
    RedirectWindow(display=self.display, opcode=self.display.get_extension_major(extname), window=self, update=update)