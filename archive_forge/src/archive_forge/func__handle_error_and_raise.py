from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
from apitools.base.py import http_wrapper as apitools_http_wrapper
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
def _handle_error_and_raise(retry_args):
    apitools_http_wrapper.HandleExceptionsAndRebuildHttpConnections(retry_args)
    if isinstance(retry_args.exc, OSError) and retry_args.exc.errno == errno.ENOSPC:
        raise retry_args.exc
    raise errors.RetryableApiError()