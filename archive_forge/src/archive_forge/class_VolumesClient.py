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
class VolumesClient(object):
    """Wrapper for working with Storage Pool in the Cloud NetApp Files API Client.
  """

    def __init__(self, release_track=base.ReleaseTrack.ALPHA):
        self.release_track = release_track
        if self.release_track == base.ReleaseTrack.ALPHA:
            self._adapter = AlphaVolumesAdapter()
        elif self.release_track == base.ReleaseTrack.BETA:
            self._adapter = BetaVolumesAdapter()
        elif self.release_track == base.ReleaseTrack.GA:
            self._adapter = VolumesAdapter()
        else:
            raise ValueError('[{}] is not a valid API version.'.format(util.VERSION_MAP[release_track]))

    @property
    def client(self):
        return self._adapter.client

    @property
    def messages(self):
        return self._adapter.messages

    def WaitForOperation(self, operation_ref):
        """Waits on the long-running operation until the done field is True.

    Args:
      operation_ref: the operation reference.

    Raises:
      waiter.OperationError: if the operation contains an error.

    Returns:
      the 'response' field of the Operation.
    """
        return waiter.WaitFor(waiter.CloudOperationPollerNoResources(self.client.projects_locations_operations), operation_ref, 'Waiting for [{0}] to finish'.format(operation_ref.Name()))

    def ListVolumes(self, location_ref, limit=None):
        """Make API calls to List active Cloud NetApp Volumes.

    Args:
      location_ref: The parsed location of the listed NetApp Volumes.
      limit: The number of Cloud NetApp Volumes to limit the results to. This
        limit is passed to the server and the server does the limiting.

    Returns:
      Generator that yields the Cloud NetApp Volumes.
    """
        request = self.messages.NetappProjectsLocationsVolumesListRequest(parent=location_ref)
        response = self.client.projects_locations_volumes.List(request)
        for location in response.unreachable:
            log.warning('Location {} may be unreachable.'.format(location))
        return list_pager.YieldFromList(self.client.projects_locations_volumes, request, field=constants.VOLUME_RESOURCE, limit=limit, batch_size_attribute='pageSize')

    def CreateVolume(self, volume_ref, async_, config):
        """Create a Cloud NetApp Volume."""
        request = self.messages.NetappProjectsLocationsVolumesCreateRequest(parent=volume_ref.Parent().RelativeName(), volumeId=volume_ref.Name(), volume=config)
        create_op = self.client.projects_locations_volumes.Create(request)
        if async_:
            return create_op
        operation_ref = resources.REGISTRY.ParseRelativeName(create_op.name, collection=constants.OPERATIONS_COLLECTION)
        return self.WaitForOperation(operation_ref)

    def ParseVolumeConfig(self, name=None, capacity=None, description=None, storage_pool=None, protocols=None, share_name=None, export_policy=None, unix_permissions=None, smb_settings=None, snapshot_policy=None, snap_reserve=None, snapshot_directory=None, security_style=None, enable_kerberos=None, snapshot=None, backup=None, restricted_actions=None, backup_config=None, large_capacity=None, multiple_endpoints=None, tiering_policy=None, labels=None):
        """Parses the command line arguments for Create Volume into a config."""
        return self._adapter.ParseVolumeConfig(name=name, capacity=capacity, description=description, storage_pool=storage_pool, protocols=protocols, share_name=share_name, export_policy=export_policy, unix_permissions=unix_permissions, smb_settings=smb_settings, snapshot_policy=snapshot_policy, snap_reserve=snap_reserve, snapshot_directory=snapshot_directory, security_style=security_style, enable_kerberos=enable_kerberos, snapshot=snapshot, backup=backup, restricted_actions=restricted_actions, backup_config=backup_config, large_capacity=large_capacity, multiple_endpoints=multiple_endpoints, tiering_policy=tiering_policy, labels=labels)

    def GetVolume(self, volume_ref):
        """Get Cloud NetApp Volume information."""
        request = self.messages.NetappProjectsLocationsVolumesGetRequest(name=volume_ref.RelativeName())
        return self.client.projects_locations_volumes.Get(request)

    def DeleteVolume(self, volume_ref, async_, force):
        """Deletes an existing Cloud NetApp Volume."""
        request = self.messages.NetappProjectsLocationsVolumesDeleteRequest(name=volume_ref.RelativeName(), force=force)
        return self._DeleteVolume(async_, request)

    def _DeleteVolume(self, async_, request):
        delete_op = self.client.projects_locations_volumes.Delete(request)
        if async_:
            return delete_op
        operation_ref = resources.REGISTRY.ParseRelativeName(delete_op.name, collection=constants.OPERATIONS_COLLECTION)
        return self.WaitForOperation(operation_ref)

    def RevertVolume(self, volume_ref, snapshot_id, async_):
        """Reverts an existing Cloud NetApp Volume."""
        request = self.messages.NetappProjectsLocationsVolumesRevertRequest(name=volume_ref.RelativeName(), revertVolumeRequest=self.messages.RevertVolumeRequest(snapshotId=snapshot_id))
        revert_op = self.client.projects_locations_volumes.Revert(request)
        if async_:
            return revert_op
        operation_ref = resources.REGISTRY.ParseRelativeName(revert_op.name, collection=constants.OPERATIONS_COLLECTION)
        return self.WaitForOperation(operation_ref)

    def ParseUpdatedVolumeConfig(self, volume_config, description=None, labels=None, storage_pool=None, protocols=None, share_name=None, export_policy=None, capacity=None, unix_permissions=None, smb_settings=None, snapshot_policy=None, snap_reserve=None, snapshot_directory=None, security_style=None, enable_kerberos=None, snapshot=None, backup=None, restricted_actions=None, backup_config=None, tiering_policy=None):
        """Parses updates into a volume config."""
        return self._adapter.ParseUpdatedVolumeConfig(volume_config, description=description, labels=labels, storage_pool=storage_pool, protocols=protocols, share_name=share_name, export_policy=export_policy, capacity=capacity, unix_permissions=unix_permissions, smb_settings=smb_settings, snapshot_policy=snapshot_policy, snap_reserve=snap_reserve, snapshot_directory=snapshot_directory, security_style=security_style, enable_kerberos=enable_kerberos, snapshot=snapshot, backup=backup, restricted_actions=restricted_actions, backup_config=backup_config, tiering_policy=tiering_policy)

    def UpdateVolume(self, volume_ref, volume_config, update_mask, async_):
        """Updates a Cloud NetApp Volume.

    Args:
      volume_ref: the reference to the Volume.
      volume_config: Volume config, the updated volume.
      update_mask: str, a comma-separated list of updated fields.
      async_: bool, if False, wait for the operation to complete.

    Returns:
      an Operation or Volume message.
    """
        update_op = self._adapter.UpdateVolume(volume_ref, volume_config, update_mask)
        if async_:
            return update_op
        operation_ref = resources.REGISTRY.ParseRelativeName(update_op.name, collection=constants.OPERATIONS_COLLECTION)
        return self.WaitForOperation(operation_ref)