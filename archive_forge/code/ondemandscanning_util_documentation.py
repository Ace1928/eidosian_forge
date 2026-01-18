from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ondemandscanning import util as ods_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import resources
Silently waits for the given google.longrunning.Operation to complete.

  Args:
    operation: The operation to poll.
    version: The ODS API version endpoints to use to talk to the Operations
      service.

  Raises:
    apitools.base.py.HttpError: if the request returns an HTTP error

  Returns:
    The response field of the completed operation.
  