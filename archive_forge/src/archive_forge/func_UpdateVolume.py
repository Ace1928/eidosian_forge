from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def UpdateVolume(self, volume_ref, volume_config, update_mask):
    """Send a Patch request for the Cloud NetApp Volume."""
    update_request = self.messages.NetappProjectsLocationsVolumesPatchRequest(volume=volume_config, name=volume_ref.RelativeName(), updateMask=update_mask)
    update_op = self.client.projects_locations_volumes.Patch(update_request)
    return update_op