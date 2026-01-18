from __future__ import (absolute_import, division, print_function)
import os
import time
import glob
from abc import ABCMeta, abstractmethod
from ansible.module_utils.six import with_metaclass
def wait_for_lock(self):
    """Poll until the lock is removed if timeout is a positive number"""
    if not self._is_lockfile_present():
        return
    if self.lock_timeout > 0:
        for iteration in range(0, self.lock_timeout):
            time.sleep(1)
            if not self._is_lockfile_present():
                return
    self.module.fail_json(msg='{0} lockfile is held by another process'.format(self.pkg_mgr_name))