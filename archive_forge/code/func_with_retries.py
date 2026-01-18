from distutils.version import StrictVersion
import functools
from http import client as http_client
import json
import logging
import re
import textwrap
import time
from urllib import parse as urlparse
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common.i18n import _
from ironicclient import exc
def with_retries(func):
    """Wrapper for _http_request adding support for retries."""

    @functools.wraps(func)
    def wrapper(self, url, method, **kwargs):
        if self.conflict_max_retries is None:
            self.conflict_max_retries = DEFAULT_MAX_RETRIES
        if self.conflict_retry_interval is None:
            self.conflict_retry_interval = DEFAULT_RETRY_INTERVAL
        num_attempts = self.conflict_max_retries + 1
        for attempt in range(1, num_attempts + 1):
            try:
                return func(self, url, method, **kwargs)
            except _RETRY_EXCEPTIONS as error:
                msg = 'Error contacting Ironic server: %(error)s. Attempt %(attempt)d of %(total)d' % {'attempt': attempt, 'total': num_attempts, 'error': error}
                if attempt == num_attempts:
                    LOG.error(msg)
                    raise
                else:
                    LOG.debug(msg)
                    time.sleep(self.conflict_retry_interval)
    return wrapper