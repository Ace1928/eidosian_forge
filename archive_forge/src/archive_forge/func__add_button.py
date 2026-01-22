import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def _add_button(self, button, group, position):
    if group not in self._groups:
        if self._groups:
            self._add_separator()
        group_box = Gtk.Box()
        self._tool_box.append(group_box)
        self._groups[group] = group_box
    self._groups[group].insert_child_after(button, self._find_child_at_position(group, position))