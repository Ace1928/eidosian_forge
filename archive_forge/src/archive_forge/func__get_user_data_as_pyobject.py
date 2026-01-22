import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def _get_user_data_as_pyobject(iter):
    citer = _CTreeIter.from_iter(iter)
    return ctypes.cast(citer.contents.user_data, ctypes.py_object).value