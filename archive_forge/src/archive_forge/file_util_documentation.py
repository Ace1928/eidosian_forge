import errno
import os
import pwd
import shutil
import stat
import tempfile
Find the home directory of a user.

  Args:
    user: int, str, or None - the uid or login of the user to query for,
          or None (the default) to query for the current process' effective user

  Returns:
    str - the user's home directory

  Raises:
    TypeError: if user is not int, str, or None.
  