from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class TPUInMaintenanceEvent(exceptions.Error):
    """Error when TPU has unhealthy maintenance for health."""

    def __init__(self):
        super(TPUInMaintenanceEvent, self).__init__('This TPU is going through a maintenance event, and is currently unavailable. For more information, see https://cloud.google.com/tpu/docs/maintenance-events.')