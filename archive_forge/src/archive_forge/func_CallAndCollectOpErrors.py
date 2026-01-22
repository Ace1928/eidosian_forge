from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
import enum
from googlecloudsdk.api_lib.app import exceptions as app_exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import requests
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
import six
def CallAndCollectOpErrors(method, *args, **kwargs):
    """Wrapper for method(...) which re-raises operation-style errors.

  Args:
    method: Original method to call.
    *args: Positional arguments to method.
    **kwargs: Keyword arguments to method.

  Raises:
    MiscOperationError: If the method call itself raises one of the exceptions
      listed below. Otherwise, the original exception is raised. Preserves
      stack trace. Re-uses the error string from original error or in the case
      of HttpError, we synthesize human-friendly string from HttpException.
      However, HttpException is neither raised nor part of the stack trace.

  Returns:
    Result of calling method(*args, **kwargs).
  """
    try:
        return method(*args, **kwargs)
    except apitools_exceptions.HttpError as http_err:
        _ReraiseMiscOperationError(api_exceptions.HttpException(http_err))
    except (OperationError, OperationTimeoutError, app_exceptions.Error) as err:
        _ReraiseMiscOperationError(err)