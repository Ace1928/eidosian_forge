from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
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