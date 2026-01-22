from __future__ import annotations
from random import randrange
from typing import Any, Callable, TextIO, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import interfaces, utils
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
def _removeSafely(path):
    """
    Safely remove a path, recursively.

    If C{path} does not contain a node named C{_trial_marker}, a
    L{_NoTrialMarker} exception is raised and the path is not removed.
    """
    if not path.child(b'_trial_marker').exists():
        raise _NoTrialMarker(f'{path!r} is not a trial temporary path, refusing to remove it')
    try:
        path.remove()
    except OSError as e:
        print('could not remove %r, caught OSError [Errno %s]: %s' % (path, e.errno, e.strerror))
        try:
            newPath = FilePath(b'_trial_temp_old' + str(randrange(10000000)).encode('utf-8'))
            path.moveTo(newPath)
        except OSError as e:
            print('could not rename path, caught OSError [Errno %s]: %s' % (e.errno, e.strerror))
            raise