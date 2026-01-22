import re
from . import decorator
def _squash_name(self, name):
    """Return vfat-squashed filename.

        The name is returned as it will be stored on disk.  This raises an
        error if there are invalid characters in the name.
        """
    if re.search('[?*:;<>]', name):
        raise ValueError('illegal characters for VFAT filename: %r' % name)
    return name.lower()