import os
from .... import errors
from .... import transport as _mod_transport
from ....bzr import versionedfile
from ....errors import BzrError, UnlistableStore
from ....trace import mutter
def total_size(self):
    """Return (count, bytes)

        This is the (compressed) size stored on disk, not the size of
        the content."""
    total = 0
    count = 0
    for relpath in self._transport.iter_files_recursive():
        count += 1
        total += self._transport.stat(relpath).st_size
    return (count, total)