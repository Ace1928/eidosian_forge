import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def _patch_module(name, new_value):
    old_value = sys.modules.get(name, _unset)
    sys.modules[name] = new_value
    _module_patches.append((name, old_value))