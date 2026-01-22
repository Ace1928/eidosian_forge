import os
import stat
from breezy import errors, osutils
class AtomicFileAlreadyClosed(errors.PathError):
    _fmt = '"%(function)s" called on an AtomicFile after it was closed: "%(path)s"'

    def __init__(self, path, function):
        errors.PathError.__init__(self, path=path, extra=None)
        self.function = function