import builtins
import types
import sys
from inspect import getmembers
from webob.exc import HTTPFound
from .util import iscontroller, _cfg
def walk_controller(root_class, controller, hooks, seen=None):
    seen = seen or set()
    if type(controller) not in vars(builtins).values():
        try:
            if controller in seen:
                return
            seen.add(controller)
        except TypeError:
            return
        for hook in getattr(controller, '__hooks__', []):
            hooks.add(hook)
        for name, value in getmembers(controller):
            if name == 'controller':
                continue
            if name.startswith('__') and name.endswith('__'):
                continue
            if iscontroller(value):
                for hook in hooks:
                    value._pecan.setdefault('hooks', set()).add(hook)
            elif hasattr(value, '__class__'):
                if isinstance(value, types.MethodType) and any(filter(lambda c: value.__func__ in c.__dict__.values(), value.__self__.__class__.mro()[1:])):
                    continue
                walk_controller(root_class, value, hooks, seen)