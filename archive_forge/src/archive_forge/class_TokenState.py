import abc
from enum import Enum
import os
from google.auth import _helpers, environment_vars
from google.auth import exceptions
from google.auth import metrics
from google.auth._refresh_worker import RefreshThreadManager
class TokenState(Enum):
    """
    Tracks the state of a token.
    FRESH: The token is valid. It is not expired or close to expired, or the token has no expiry.
    STALE: The token is close to expired, and should be refreshed. The token can be used normally.
    INVALID: The token is expired or invalid. The token cannot be used for a normal operation.
    """
    FRESH = 1
    STALE = 2
    INVALID = 3