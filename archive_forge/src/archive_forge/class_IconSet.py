import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class IconSet(Gtk.IconSet):

    def __new__(cls, pixbuf=None):
        if pixbuf is not None:
            warnings.warn('Gtk.IconSet(pixbuf) has been deprecated. Please use: Gtk.IconSet.new_from_pixbuf(pixbuf)', PyGTKDeprecationWarning, stacklevel=2)
            iconset = Gtk.IconSet.new_from_pixbuf(pixbuf)
        else:
            iconset = Gtk.IconSet.__new__(cls)
        return iconset

    def __init__(self, *args, **kwargs):
        return super(IconSet, self).__init__()