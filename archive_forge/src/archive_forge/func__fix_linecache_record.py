import builtins
import inspect
import itertools
import linecache
import sys
import threading
import types
from tensorflow.python.util import tf_inspect
def _fix_linecache_record(obj):
    """Fixes potential corruption of linecache in the presence of functools.wraps.

  functools.wraps modifies the target object's __module__ field, which seems
  to confuse linecache in special instances, for example when the source is
  loaded from a .par file (see https://google.github.io/subpar/subpar.html).

  This function simply triggers a call to linecache.updatecache when a mismatch
  was detected between the object's __module__ property and the object's source
  file.

  Args:
    obj: Any
  """
    if hasattr(obj, '__module__'):
        obj_file = inspect.getfile(obj)
        obj_module = obj.__module__
        loaded_modules = tuple(sys.modules.values())
        for m in loaded_modules:
            if hasattr(m, '__file__') and m.__file__ == obj_file:
                if obj_module is not m:
                    linecache.updatecache(obj_file, m.__dict__)