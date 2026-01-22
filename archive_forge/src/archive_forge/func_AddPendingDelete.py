from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
def AddPendingDelete(self, environment_name, operation):
    """Adds an environment whose deletion to track.

    Args:
      environment_name: str, the relative resource name of the environment
          being deleted
      operation: Operation, the longrunning operation object returned by the
          API when the deletion was initiated
    """
    self.pending_deletes.append(_PendingEnvironmentDelete(environment_name, operation))