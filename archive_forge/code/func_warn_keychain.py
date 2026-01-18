import platform
import os
import warnings
import functools
from ...backend import KeyringBackend
from ...errors import PasswordSetError
from ...errors import PasswordDeleteError
from ...errors import KeyringLocked
from ...errors import KeyringError
from ..._compat import properties
def warn_keychain(func):

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.keychain:
            warnings.warn('Specified keychain is ignored. See #623')
        return func(self, *args, **kwargs)
    return wrapper