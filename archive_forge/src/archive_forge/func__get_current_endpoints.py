import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
def _get_current_endpoints(self, key):
    if key not in self._cache:
        return None
    now = self._time()
    return [e for e in self._cache[key] if now < e['Expiration']]