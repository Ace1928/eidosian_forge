import pickle
import random
import re
from django.core.cache.backends.base import DEFAULT_TIMEOUT, BaseCache
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
def _get_connection_pool(self, write):
    index = self._get_connection_pool_index(write)
    if index not in self._pools:
        self._pools[index] = self._pool_class.from_url(self._servers[index], **self._pool_options)
    return self._pools[index]