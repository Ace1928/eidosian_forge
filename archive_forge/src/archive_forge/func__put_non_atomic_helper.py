import errno
import os
import sys
from stat import S_IMODE, S_ISDIR, ST_MODE
from .. import osutils, transport, urlutils
def _put_non_atomic_helper(self, relpath, writer, mode=None, create_parent_dir=False, dir_mode=None):
    """Common functionality information for the put_*_non_atomic.

        This tracks all the create_parent_dir stuff.

        :param relpath: the path we are putting to.
        :param writer: A function that takes an os level file descriptor
            and writes whatever data it needs to write there.
        :param mode: The final file mode.
        :param create_parent_dir: Should we be creating the parent directory
            if it doesn't exist?
        """
    abspath = self._abspath(relpath)
    if mode is None:
        local_mode = 438
    else:
        local_mode = mode
    try:
        fd = os.open(abspath, _put_non_atomic_flags, local_mode)
    except OSError as e:
        if not create_parent_dir or e.errno not in (errno.ENOENT, errno.ENOTDIR):
            self._translate_error(e, relpath)
        parent_dir = os.path.dirname(abspath)
        if not parent_dir:
            self._translate_error(e, relpath)
        self._mkdir(parent_dir, mode=dir_mode)
        try:
            fd = os.open(abspath, _put_non_atomic_flags, local_mode)
        except OSError as e:
            self._translate_error(e, relpath)
    try:
        st = os.fstat(fd)
        if mode is not None and mode != S_IMODE(st.st_mode):
            osutils.chmod_if_possible(abspath, mode)
        writer(fd)
    finally:
        os.close(fd)