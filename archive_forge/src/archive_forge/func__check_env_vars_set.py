import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def _check_env_vars_set(self, dir_env_var, file_env_var):
    """
        Check to see if the default cert dir/file environment vars are present.

        :return: bool
        """
    return os.environ.get(file_env_var) is not None or os.environ.get(dir_env_var) is not None