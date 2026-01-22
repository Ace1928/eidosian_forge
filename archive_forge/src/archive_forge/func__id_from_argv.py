import sys
import os
import contextlib
from ..backend import KeyringBackend
from ..credentials import SimpleCredential
from ..errors import PasswordDeleteError
from ..errors import PasswordSetError, InitError, KeyringLocked
from .._compat import properties
def _id_from_argv():
    """
    Safely infer an app id from sys.argv.
    """
    allowed = (AttributeError, IndexError, TypeError)
    with contextlib.suppress(allowed):
        return sys.argv[0]