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
def HandleError(exc, command_path, known_error_handler=None):
    """Handles an error that occurs during command execution.

  It calls ConvertKnownError to convert exceptions to known types before
  processing. If it is a known type, it is printed nicely as as error. If not,
  it is raised as a crash.

  Args:
    exc: Exception, The original exception that occurred.
    command_path: str, The name of the command that failed (for error
      reporting).
    known_error_handler: f(): A function to report the current exception as a
      known error.
  """
    known_exc, print_error = ConvertKnownError(exc)
    if known_exc:
        _LogKnownError(known_exc, command_path, print_error)
        if known_error_handler:
            known_error_handler()
        if properties.VALUES.core.print_handled_tracebacks.GetBool():
            core_exceptions.reraise(exc)
        _Exit(known_exc)
    else:
        log.debug(console_attr.SafeText(exc), exc_info=sys.exc_info())
        core_exceptions.reraise(exc)