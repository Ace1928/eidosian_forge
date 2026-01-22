from __future__ import print_function
import io
import gzip
import time
from datetime import timedelta
from collections import defaultdict
import urllib3
import certifi
from sentry_sdk.utils import Dsn, logger, capture_internal_exceptions, json_dumps
from sentry_sdk.worker import BackgroundWorker
from sentry_sdk.envelope import Envelope, Item, PayloadRef
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk._types import TYPE_CHECKING
def _get_pool_options(self, ca_certs):
    options = {'num_pools': self._num_pools, 'cert_reqs': 'CERT_REQUIRED', 'ca_certs': ca_certs or certifi.where()}
    if self.options['socket_options']:
        options['socket_options'] = self.options['socket_options']
    return options