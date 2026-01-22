import time
import random
import string
import hashlib
from libcloud.utils.py3 import httplib, urlencode, basestring
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
def _salt(self):
    """Return a 16-character alphanumeric string."""
    r = random.SystemRandom()
    return ''.join((r.choice(SALT_CHARACTERS) for _ in range(16)))