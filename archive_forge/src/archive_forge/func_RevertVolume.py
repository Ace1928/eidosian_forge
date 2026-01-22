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
def RevertVolume(self, volume_ref, snapshot_id, async_):
    """Reverts an existing Cloud NetApp Volume."""
    request = self.messages.NetappProjectsLocationsVolumesRevertRequest(name=volume_ref.RelativeName(), revertVolumeRequest=self.messages.RevertVolumeRequest(snapshotId=snapshot_id))
    revert_op = self.client.projects_locations_volumes.Revert(request)
    if async_:
        return revert_op
    operation_ref = resources.REGISTRY.ParseRelativeName(revert_op.name, collection=constants.OPERATIONS_COLLECTION)
    return self.WaitForOperation(operation_ref)