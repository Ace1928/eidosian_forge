from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.core.exceptions import Error
def GetDiskRequestMessage(self):
    return self._messages.ComputeRegionDisksGetRequest(**self._disk_ref.AsDict())