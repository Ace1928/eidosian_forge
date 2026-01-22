from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_fusion import datafusion as df
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions as core_exceptions
class OperationError(core_exceptions.Error):
    """Class for errors raised when a polled operation completes with an error."""

    def __init__(self, operation_name, description):
        super(OperationError, self).__init__('Operation [{}] failed: {}'.format(operation_name, description))