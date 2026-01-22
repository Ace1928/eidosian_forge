import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
def _refresh_current_endpoints(self, **kwargs):
    cache_key = self._create_cache_key(**kwargs)
    try:
        response = self._describe_endpoints(**kwargs)
        endpoints = self._parse_endpoints(response)
        self._cache[cache_key] = endpoints
        self._failed_attempts.pop(cache_key, None)
        return endpoints
    except (ConnectionError, HTTPClientError):
        self._failed_attempts[cache_key] = self._time() + 60
        return None