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
def _check_safe(self):
    if self.pid != os.getpid():
        with self._lock:
            if self.pid == os.getpid():
                return
            self.disconnect_all()
            self.reset()