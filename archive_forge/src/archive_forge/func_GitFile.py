import os
import sys
import warnings
from typing import ClassVar, Set
def GitFile(filename, mode='rb', bufsize=-1, mask=420):
    """Create a file object that obeys the git file locking protocol.

    Returns: a builtin file object or a _GitFile object

    Note: See _GitFile for a description of the file locking protocol.

    Only read-only and write-only (binary) modes are supported; r+, w+, and a
    are not.  To read and write from the same file, you can take advantage of
    the fact that opening a file for write does not actually open the file you
    request.

    The default file mask makes any created files user-writable and
    world-readable.

    """
    if 'a' in mode:
        raise OSError('append mode not supported for Git files')
    if '+' in mode:
        raise OSError('read/write mode not supported for Git files')
    if 'b' not in mode:
        raise OSError('text mode not supported for Git files')
    if 'w' in mode:
        return _GitFile(filename, mode, bufsize, mask)
    else:
        return open(filename, mode, bufsize)