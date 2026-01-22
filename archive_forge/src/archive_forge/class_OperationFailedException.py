from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from apitools.base.py import encoding
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker as tracker
from googlecloudsdk.core.util import retry
class OperationFailedException(core_exceptions.Error):

    def __init__(self, operation_with_error):
        op_id = OperationNameToId(operation_with_error.name)
        error_code = operation_with_error.error.code
        error_message = operation_with_error.error.message
        message = 'Operation [{0}] failed: {1}: {2}'.format(op_id, error_code, error_message)
        super(OperationFailedException, self).__init__(message)