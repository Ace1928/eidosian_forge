import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def WriteOnDiff(filename):
    """Write to a file only if the new contents differ.

  Arguments:
    filename: name of the file to potentially write to.
  Returns:
    A file like object which will write to temporary file and only overwrite
    the target if it differs (on close).
  """

    class Writer:
        """Wrapper around file which only covers the target if it differs."""

        def __init__(self):
            base_temp_dir = '' if IsCygwin() else os.path.dirname(filename)
            tmp_fd, self.tmp_path = tempfile.mkstemp(suffix='.tmp', prefix=os.path.split(filename)[1] + '.gyp.', dir=base_temp_dir)
            try:
                self.tmp_file = os.fdopen(tmp_fd, 'wb')
            except Exception:
                os.unlink(self.tmp_path)
                raise

        def __getattr__(self, attrname):
            return getattr(self.tmp_file, attrname)

        def close(self):
            try:
                self.tmp_file.close()
                same = False
                try:
                    same = filecmp.cmp(self.tmp_path, filename, False)
                except OSError as e:
                    if e.errno != errno.ENOENT:
                        raise
                if same:
                    os.unlink(self.tmp_path)
                else:
                    umask = os.umask(63)
                    os.umask(umask)
                    os.chmod(self.tmp_path, 438 & ~umask)
                    if sys.platform == 'win32' and os.path.exists(filename):
                        os.remove(filename)
                    os.rename(self.tmp_path, filename)
            except Exception:
                os.unlink(self.tmp_path)
                raise

        def write(self, s):
            self.tmp_file.write(s.encode('utf-8'))
    return Writer()