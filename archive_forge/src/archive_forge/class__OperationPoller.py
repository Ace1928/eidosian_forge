from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
class _OperationPoller(waiter.CloudOperationPollerNoResources):
    """ Class for polling Composer longrunning Operations. """

    def __init__(self, release_track=base.ReleaseTrack.GA):
        super(_OperationPoller, self).__init__(GetService(release_track=release_track), lambda x: x)

    def IsDone(self, operation):
        if operation.done:
            if operation.error:
                raise command_util.OperationError(operation.name, operation.error.message)
            return True
        return False