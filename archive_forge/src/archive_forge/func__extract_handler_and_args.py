import os
from collections import abc
from functools import partial
from gi.repository import GLib, GObject, Gio
def _extract_handler_and_args(obj_or_map, handler_name):
    handler = None
    if isinstance(obj_or_map, abc.Mapping):
        handler = obj_or_map.get(handler_name, None)
    else:
        handler = getattr(obj_or_map, handler_name, None)
    if handler is None:
        raise AttributeError('Handler %s not found' % handler_name)
    args = ()
    if isinstance(handler, abc.Sequence):
        if len(handler) == 0:
            raise TypeError('Handler %s tuple can not be empty' % handler)
        args = handler[1:]
        handler = handler[0]
    elif not callable(handler):
        raise TypeError('Handler %s is not a method, function or tuple' % handler)
    return (handler, args)