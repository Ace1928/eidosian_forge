import re
from tensorflow.python.util import tf_inspect
def _is_private(self, path, name, obj=None):
    """Return whether a name is private."""
    del obj
    return path in self._private_map and name in self._private_map[path] or (name.startswith('_') and (not re.match('__.*__$', name)) or name in ['__base__', '__class__', '__next_in_mro__'])