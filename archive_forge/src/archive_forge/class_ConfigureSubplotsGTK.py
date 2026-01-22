import logging
import sys
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backend_tools import Cursors
import gi
from gi.repository import Gdk, Gio, GLib, Gtk
class ConfigureSubplotsGTK(backend_tools.ConfigureSubplotsBase):

    def trigger(self, *args):
        _NavigationToolbar2GTK.configure_subplots(self, None)