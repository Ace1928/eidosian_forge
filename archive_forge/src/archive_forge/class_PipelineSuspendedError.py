from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class PipelineSuspendedError(exceptions.Error):
    """Error when a user performs an activity on a suspended pipeline."""

    def __init__(self, pipeline_name, failed_activity_msg):
        error_msg = '{} DeliveryPipeline {} is suspended.'.format(failed_activity_msg, pipeline_name)
        super(PipelineSuspendedError, self).__init__(error_msg)