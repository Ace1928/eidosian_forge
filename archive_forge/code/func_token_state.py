import abc
from enum import Enum
import os
from google.auth import _helpers, environment_vars
from google.auth import exceptions
from google.auth import metrics
from google.auth._refresh_worker import RefreshThreadManager
@property
def token_state(self):
    """
        See `:obj:`TokenState`
        """
    if self.token is None:
        return TokenState.INVALID
    if self.expiry is None:
        return TokenState.FRESH
    expired = _helpers.utcnow() >= self.expiry
    if expired:
        return TokenState.INVALID
    is_stale = _helpers.utcnow() >= self.expiry - _helpers.REFRESH_THRESHOLD
    if is_stale:
        return TokenState.STALE
    return TokenState.FRESH