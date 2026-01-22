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
def ParseUpdatedVolumeConfig(self, volume_config, description=None, labels=None, storage_pool=None, protocols=None, share_name=None, export_policy=None, capacity=None, unix_permissions=None, smb_settings=None, snapshot_policy=None, snap_reserve=None, snapshot_directory=None, security_style=None, enable_kerberos=None, active_directory=None, snapshot=None, backup=None, restricted_actions=None, backup_config=None, large_capacity=None, multiple_endpoints=None, tiering_policy=None):
    """Parse update information into an updated Volume message."""
    if description is not None:
        volume_config.description = description
    if labels is not None:
        volume_config.labels = labels
    if capacity is not None:
        volume_config.capacityGib = capacity
    if storage_pool is not None:
        volume_config.storagePool = storage_pool
    if protocols is not None:
        self.ParseProtocols(volume_config, protocols)
    if share_name is not None:
        volume_config.shareName = share_name
    if export_policy is not None:
        self.ParseExportPolicy(volume_config, export_policy)
    if unix_permissions is not None:
        volume_config.unixPermissions = unix_permissions
    if smb_settings is not None:
        volume_config.smbSettings = smb_settings
    if snapshot_policy is not None:
        self.ParseSnapshotPolicy(volume_config, snapshot_policy)
    if snap_reserve is not None:
        volume_config.snapReserve = snap_reserve
    if snapshot_directory is not None:
        volume_config.snapshotDirectory = snapshot_directory
    if security_style is not None:
        volume_config.securityStyle = security_style
    if enable_kerberos is not None:
        volume_config.kerberosEnabled = enable_kerberos
    if active_directory is not None:
        volume_config.activeDirectory = active_directory
    if snapshot is not None or backup is not None:
        self.ParseRestoreParameters(volume_config, snapshot, backup)
    if restricted_actions is not None:
        volume_config.restrictedActions = restricted_actions
    if backup_config is not None:
        self.ParseBackupConfig(volume_config, backup_config)
    if large_capacity is not None:
        volume_config.largeCapacity = large_capacity
    if multiple_endpoints is not None:
        volume_config.multipleEndpoints = multiple_endpoints
    if tiering_policy is not None:
        self.ParseTieringPolicy(volume_config, tiering_policy)
    return volume_config