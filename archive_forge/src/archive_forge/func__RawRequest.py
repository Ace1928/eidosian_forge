from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import stat
import tarfile
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import local_file_adapter
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import http_encoding
from googlecloudsdk.core.util import retry
import requests
import six
def _RawRequest(*args, **kwargs):
    """Executes an HTTP request."""

    def RetryIf(exc_type, exc_value, unused_traceback, unused_state):
        return exc_type == requests.exceptions.HTTPError and exc_value.response.status_code == 404

    def StatusUpdate(unused_result, unused_state):
        log.debug('Retrying request...')
    retryer = retry.Retryer(max_retrials=3, exponential_sleep_multiplier=2, jitter_ms=100, status_update_func=StatusUpdate)
    try:
        return retryer.RetryOnException(_ExecuteRequestAndRaiseExceptions, args, kwargs, should_retry_if=RetryIf, sleep_ms=500)
    except retry.RetryException as e:
        if e.last_result[1]:
            exceptions.reraise(e.last_result[1][1], tb=e.last_result[1][2])
        raise