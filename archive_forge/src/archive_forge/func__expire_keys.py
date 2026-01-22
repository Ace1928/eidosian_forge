from __future__ import (absolute_import, division, print_function)
import collections
import os
import time
from multiprocessing import Lock
from itertools import chain
from ansible.errors import AnsibleError
from ansible.module_utils.common._collections_compat import MutableSet
from ansible.plugins.cache import BaseCacheModule
from ansible.utils.display import Display
def _expire_keys(self):
    if self._timeout > 0:
        expiry_age = time.time() - self._timeout
        self._keys.remove_by_timerange(0, expiry_age)