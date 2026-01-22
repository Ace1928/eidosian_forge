import os
import io
from .._utils import set_module
def _fullpath(self, path):
    """Return complete path for path.  Prepends baseurl if necessary."""
    splitpath = path.split(self._baseurl, 2)
    if len(splitpath) == 1:
        result = os.path.join(self._baseurl, path)
    else:
        result = path
    return result