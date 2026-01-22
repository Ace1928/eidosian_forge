from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
import frozendict
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util as projects_api_util
from googlecloudsdk.api_lib.functions.v2 import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import encoding as encoder
from googlecloudsdk.core.util import retry
import six
def OperationErrorToString(error):
    """Returns a human readable string representation from the operation.

  Args:
    error: A string representing the raw json of the operation error.

  Returns:
    A human readable string representation of the error.
  """
    error_message = 'OperationError: code={0}, message={1}'.format(error.code, encoder.Decode(error.message))
    messages = apis.GetMessagesModule('cloudfunctions', _V2_ALPHA)
    if error.details:
        for detail in error.details:
            sub_error = encoding.PyValueToMessage(messages.Status, encoding.MessageToPyValue(detail))
            if sub_error.code is not None or sub_error.message is not None:
                error_message += '\n' + OperationErrorToString(sub_error)
    return error_message