import logging
from .._compat import properties
from ..backend import KeyringBackend
from ..credentials import SimpleCredential
from ..errors import PasswordDeleteError, ExceptionRaisedContext

        If available, the preferred backend on Windows.
        