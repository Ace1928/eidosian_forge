from xcffib import visualtype_to_c_struct
from . import cairo, constants
from .surfaces import SURFACE_TYPE_TO_CLASS, Surface

        Informs cairo of the new size of the X Drawable underlying the surface.
        For a surface created for a Window (rather than a Pixmap), this
        function must be called each time the size of the window changes (for
        a subwindow, you are normally resizing the window yourself, but for a
        toplevel window, it is necessary to listen for
        :class:`xcffib.xproto.ConfigureNotifyEvent`'s).

        A Pixmap can never change size, so it is never necessary to call this
        function on a surface created for a Pixmap.

        :param width: integer
        :param height: integer
        