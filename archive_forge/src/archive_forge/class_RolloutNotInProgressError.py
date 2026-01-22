from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class RolloutNotInProgressError(exceptions.Error):
    """Error when a rollout is not in_progress, but is expected to be."""

    def __init__(self, rollout_name):
        super(RolloutNotInProgressError, self).__init__('Rollout {} is not IN_PROGRESS.'.format(rollout_name))