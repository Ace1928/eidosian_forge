import logging
from .._compat import properties
from ..backend import KeyringBackend
from ..credentials import SimpleCredential
from ..errors import PasswordDeleteError, ExceptionRaisedContext
@staticmethod
def _compound_name(username, service):
    return f'{username}@{service}'