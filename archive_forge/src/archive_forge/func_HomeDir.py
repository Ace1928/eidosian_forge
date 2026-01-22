import errno
import os
import pwd
import shutil
import stat
import tempfile
def HomeDir(user=None):
    """Find the home directory of a user.

  Args:
    user: int, str, or None - the uid or login of the user to query for,
          or None (the default) to query for the current process' effective user

  Returns:
    str - the user's home directory

  Raises:
    TypeError: if user is not int, str, or None.
  """
    if user is None:
        pw_struct = pwd.getpwuid(os.geteuid())
    elif isinstance(user, int):
        pw_struct = pwd.getpwuid(user)
    elif isinstance(user, str):
        pw_struct = pwd.getpwnam(user)
    else:
        raise TypeError('user must be None or an instance of int or str')
    return pw_struct.pw_dir