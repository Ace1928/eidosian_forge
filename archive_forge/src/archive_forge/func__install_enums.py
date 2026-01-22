import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def _install_enums(module, dest=None, strip=''):
    if dest is None:
        dest = module
    modname = dest.__name__.rsplit('.', 1)[1].upper()
    for attr in dir(module):
        try:
            obj = getattr(module, attr, None)
        except:
            continue
        try:
            if issubclass(obj, GObject.GEnum):
                for value, enum in obj.__enum_values__.items():
                    name = enum.value_name
                    name = name.replace(modname + '_', '')
                    if strip and name.startswith(strip):
                        name = name[len(strip):]
                    _patch(dest, name, enum)
        except TypeError:
            continue
        try:
            if issubclass(obj, GObject.GFlags):
                for value, flag in obj.__flags_values__.items():
                    try:
                        name = flag.value_names[-1].replace(modname + '_', '')
                    except IndexError:
                        continue
                    _patch(dest, name, flag)
        except TypeError:
            continue