import json
import os
from collections import defaultdict
import hashlib
import tempfile
from functools import partial
import kubernetes.dynamic
import kubernetes.dynamic.discovery
from kubernetes import __version__
from kubernetes.dynamic.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.client.resource import (
def __init_cache(self, refresh=False):
    if refresh or not os.path.exists(self.__cache_file):
        self._cache = {'library_version': __version__}
        refresh = True
    else:
        try:
            with open(self.__cache_file, 'r') as f:
                self._cache = json.load(f, cls=partial(CacheDecoder, self.client))
            if self._cache.get('library_version') != __version__:
                self.invalidate_cache()
        except Exception:
            self.invalidate_cache()
    self._load_server_info()
    self.discover()
    if refresh:
        self._write_cache()