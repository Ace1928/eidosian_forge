from __future__ import (absolute_import, division, print_function)
import inspect
import os
import time
from abc import ABCMeta, abstractmethod
from datetime import datetime
from ansible_collections.ovirt.ovirt.plugins.module_utils.cloud import CloudRetry
from ansible_collections.ovirt.ovirt.plugins.module_utils.version import ComparableVersion
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common._collections_compat import Mapping
def wait_for_import(self, condition=lambda e: True):
    if self._module.params['wait']:
        start = time.time()
        timeout = self._module.params['timeout']
        poll_interval = self._module.params['poll_interval']
        while time.time() < start + timeout:
            entity = self.search_entity()
            if entity and condition(entity):
                return entity
            time.sleep(poll_interval)