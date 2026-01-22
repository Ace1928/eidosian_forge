import errno
import os
import pwd
import shutil
import stat
import tempfile
def RmDirs(dir_name):
    """Removes dir_name and every non-empty directory in dir_name.

  Unlike os.removedirs and shutil.rmtree, this function doesn't raise an error
  if the directory does not exist.

  Args:
    dir_name: Directory to be removed.
  """
    try:
        shutil.rmtree(dir_name)
    except OSError as err:
        if err.errno != errno.ENOENT:
            raise
    try:
        parent_directory = os.path.dirname(dir_name)
        while parent_directory:
            try:
                os.rmdir(parent_directory)
            except OSError as err:
                if err.errno != errno.ENOENT:
                    raise
            parent_directory = os.path.dirname(parent_directory)
    except OSError as err:
        if err.errno not in (errno.EACCES, errno.ENOTEMPTY):
            raise