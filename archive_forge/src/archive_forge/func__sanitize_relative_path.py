import os
import io
from .._utils import set_module
def _sanitize_relative_path(self, path):
    """Return a sanitised relative path for which
        os.path.abspath(os.path.join(base, path)).startswith(base)
        """
    last = None
    path = os.path.normpath(path)
    while path != last:
        last = path
        path = path.lstrip(os.sep).lstrip('/')
        path = path.lstrip(os.pardir).lstrip('..')
        drive, path = os.path.splitdrive(path)
    return path