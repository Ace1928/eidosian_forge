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
def GetVolume(self, volume_ref):
    """Get Cloud NetApp Volume information."""
    request = self.messages.NetappProjectsLocationsVolumesGetRequest(name=volume_ref.RelativeName())
    return self.client.projects_locations_volumes.Get(request)