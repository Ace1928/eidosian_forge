import functools
import logging
import random
import threading
import time
from oslo_utils import excutils
from oslo_utils import importutils
from oslo_utils import reflection
from oslo_db import exception
from oslo_db import options
def _is_exception_expected(self, exc):
    if isinstance(exc, self.db_error):
        if not isinstance(exc, exception.RetryRequest):
            LOG.debug('DB error: %s', exc)
        return True
    return self.exception_checker(exc)