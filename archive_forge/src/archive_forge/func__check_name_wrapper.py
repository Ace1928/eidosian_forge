import _imp
import _io
import sys
import _warnings
import marshal
def _check_name_wrapper(self, name=None, *args, **kwargs):
    if name is None:
        name = self.name
    elif self.name != name:
        raise ImportError('loader for %s cannot handle %s' % (self.name, name), name=name)
    return method(self, name, *args, **kwargs)