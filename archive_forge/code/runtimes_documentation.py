from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
Handles Long Running Operations for both cases of async.

  Args:
    operation: The operation to poll.
    args: ArgParse instance containing user entered arguments.
    runtime_service: The service to get the resource after the long running
      operation completes.
    release_track: base.ReleaseTrack object.
    operation_type: Enum value of type OperationType indicating the kind of
      operation to wait for.

  Raises:
    apitools.base.py.HttpError: if the request returns an HTTP error

  Returns:
    The Runtime resource if synchronous, else the Operation Resource.
  