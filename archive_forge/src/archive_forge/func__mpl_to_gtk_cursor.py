import functools
import logging
import os
from pathlib import Path
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, GObject, Gtk, Gdk
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
@functools.cache
def _mpl_to_gtk_cursor(mpl_cursor):
    return Gdk.Cursor.new_from_name(Gdk.Display.get_default(), _backend_gtk.mpl_to_gtk_cursor_name(mpl_cursor))