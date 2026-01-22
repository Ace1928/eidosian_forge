import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def _mpl_coords(self, xy=None):
    """
        Convert the *xy* position of a GTK event, or of the current cursor
        position if *xy* is None, to Matplotlib coordinates.

        GTK use logical pixels, but the figure is scaled to physical pixels for
        rendering.  Transform to physical pixels so that all of the down-stream
        transforms work as expected.

        Also, the origin is different and needs to be corrected.
        """
    if xy is None:
        surface = self.get_native().get_surface()
        is_over, x, y, mask = surface.get_device_position(self.get_display().get_default_seat().get_pointer())
    else:
        x, y = xy
    x = x * self.device_pixel_ratio
    y = self.figure.bbox.height - y * self.device_pixel_ratio
    return (x, y)