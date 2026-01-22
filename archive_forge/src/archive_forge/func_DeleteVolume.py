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
def DeleteVolume(self, volume_ref, async_, force):
    """Deletes an existing Cloud NetApp Volume."""
    request = self.messages.NetappProjectsLocationsVolumesDeleteRequest(name=volume_ref.RelativeName(), force=force)
    return self._DeleteVolume(async_, request)