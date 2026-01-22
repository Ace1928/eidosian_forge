import logging
import sys
import warnings
from google.auth import version as google_auth_version
from google.auth._default import (
class Python37DeprecationWarning(DeprecationWarning):
    """
    Deprecation warning raised when Python 3.7 runtime is detected.
    Python 3.7 support will be dropped after January 1, 2024.
    """
    pass