import functools
import warnings
from collections import namedtuple
import gi.module
from gi.overrides import override, deprecated_attr
from gi.repository import GLib
from gi import PyGIDeprecationWarning
from gi import _propertyhelper as propertyhelper
from gi import _signalhelper as signalhelper
from gi import _gi
from gi import _option as option
def _unsupported_data_method(self, *args, **kargs):
    raise RuntimeError('Data access methods are unsupported. Use normal Python attributes instead')