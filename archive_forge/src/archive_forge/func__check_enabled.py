import os
from ... import urlutils
from . import request
def _check_enabled(self):
    if not vfs_enabled():
        raise request.DisabledMethod(self.__class__.__name__)