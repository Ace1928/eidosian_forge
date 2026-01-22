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
def _show_shortcuts_dialog(self):
    dialog = Gtk.MessageDialog(self._figure.canvas.get_toplevel(), 0, Gtk.MessageType.INFO, Gtk.ButtonsType.OK, self._get_help_text(), title='Help')
    dialog.run()
    dialog.destroy()