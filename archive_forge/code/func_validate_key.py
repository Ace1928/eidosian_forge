import re
import time
from django.core.cache.backends.base import (
from django.utils.functional import cached_property
def validate_key(self, key):
    for warning in memcache_key_warnings(key):
        raise InvalidCacheKey(warning)