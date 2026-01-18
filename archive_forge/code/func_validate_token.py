from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
def validate_token(self, token):
    if token == 'token':
        return 'token'
    elif token is None:
        return
    else:
        raise TokenMismatch(token, 'token')