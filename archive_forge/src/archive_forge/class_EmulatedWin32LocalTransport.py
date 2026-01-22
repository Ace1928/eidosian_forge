import errno
import os
import sys
from stat import S_IMODE, S_ISDIR, ST_MODE
from .. import osutils, transport, urlutils
class EmulatedWin32LocalTransport(LocalTransport):
    """Special transport for testing Win32 [UNC] paths on non-windows"""

    def __init__(self, base):
        if base[-1] != '/':
            base = base + '/'
        super(LocalTransport, self).__init__(base)
        self._local_base = urlutils._win32_local_path_from_url(base)

    def abspath(self, relpath):
        path = osutils._win32_normpath(osutils.pathjoin(self._local_base, urlutils.unescape(relpath)))
        return urlutils._win32_local_path_to_url(path)

    def clone(self, offset=None):
        """Return a new LocalTransport with root at self.base + offset
        Because the local filesystem does not require a connection,
        we can just return a new object.
        """
        if offset is None:
            return EmulatedWin32LocalTransport(self.base)
        else:
            abspath = self.abspath(offset)
            if abspath == 'file://':
                abspath = self.base
            return EmulatedWin32LocalTransport(abspath)