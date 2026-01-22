import types
from contextlib import contextmanager
from torch.backends import (
class ContextProp:

    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter()

    def __set__(self, obj, val):
        if not flags_frozen():
            self.setter(val)
        else:
            raise RuntimeError('not allowed to set %s flags after disable_global_flags; please use flags() context manager instead' % obj.__name__)