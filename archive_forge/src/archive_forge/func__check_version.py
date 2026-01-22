import time
import logging
import datetime
import functools
from pyzor.engines.common import *
def _check_version(self):
    """Check if there are deprecated records and warn the user."""
    old_keys = len(self.db.keys('pyzord.digest.*'))
    if old_keys:
        cmd = 'pyzor-migrate --delete --se=redis_v0 --sd=%s --de=redis --dd=%s' % (self._dsn, self._dsn)
        self.log.critical('You have %s records in the deprecated version of the redis engine.', old_keys)
        self.log.critical('Please migrate the records with: %r', cmd)