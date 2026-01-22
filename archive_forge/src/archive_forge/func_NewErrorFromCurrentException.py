from within calliope.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
from functools import wraps
import os
import sys
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
import six
def NewErrorFromCurrentException(error, *args):
    """Creates a new error based on the current exception being handled.

  If no exception is being handled, a new error with the given args
  is created.  If there is a current exception, the original exception is
  first logged (to file only).  A new error is then created with the
  same args as the current one.

  Args:
    error: The new error to create.
    *args: The standard args taken by the constructor of Exception for the new
      exception that is created.  If None, the args from the exception
      currently being handled will be used.

  Returns:
    The generated error exception.
  """
    _, current_exception, _ = sys.exc_info()
    if current_exception:
        file_logger = log.file_only_logger
        file_logger.error('Handling the source of a tool exception, original details follow.')
        file_logger.exception(current_exception)
    if args:
        return error(*args)
    elif current_exception:
        return error(*current_exception.args)
    return error('An unknown error has occurred')