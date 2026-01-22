import six
import base64
import sys
from pyu2f import errors
from pyu2f import u2f
from pyu2f.convenience import baseauthenticator
def IsAvailable(self):
    """See base class."""
    return True