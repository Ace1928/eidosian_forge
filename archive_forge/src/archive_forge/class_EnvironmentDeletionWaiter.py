from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
class EnvironmentDeletionWaiter(object):
    """Class for waiting for synchronous deletion of one or more Environments."""

    def __init__(self, release_track=base.ReleaseTrack.GA):
        self.pending_deletes = []
        self.release_track = release_track

    def AddPendingDelete(self, environment_name, operation):
        """Adds an environment whose deletion to track.

    Args:
      environment_name: str, the relative resource name of the environment
          being deleted
      operation: Operation, the longrunning operation object returned by the
          API when the deletion was initiated
    """
        self.pending_deletes.append(_PendingEnvironmentDelete(environment_name, operation))

    def Wait(self):
        """Polls pending deletions and returns when they are complete."""
        encountered_errors = False
        for pending_delete in self.pending_deletes:
            try:
                operations_api_util.WaitForOperation(pending_delete.operation, 'Waiting for [{}] to be deleted'.format(pending_delete.environment_name), release_track=self.release_track)
            except command_util.OperationError as e:
                encountered_errors = True
                log.DeletedResource(pending_delete.environment_name, kind='environment', is_async=False, failed=six.text_type(e))
        return encountered_errors