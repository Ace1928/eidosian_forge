from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class RolloutInProgressError(exceptions.Error):
    """Error when there is a rollout in progress, no to-target value is given and a promote is attempted."""

    def __init__(self, release_name, target_name):
        super(RolloutInProgressError, self).__init__('Unable to promote release {} to target {}. A rollout is already in progress.'.format(release_name, target_name))