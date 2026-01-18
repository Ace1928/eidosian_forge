from ..backend import KeyringBackend
from .._compat import properties

    Keyring that return None on every operation.

    >>> kr = Keyring()
    >>> kr.get_password('svc', 'user')
    