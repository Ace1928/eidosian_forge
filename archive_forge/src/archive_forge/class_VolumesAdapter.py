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
class VolumesAdapter(object):
    """Adapter for the Cloud NetApp Files API Volume resource."""

    def __init__(self):
        self.release_track = base.ReleaseTrack.GA
        self.client = util.GetClientInstance(release_track=self.release_track)
        self.messages = util.GetMessagesModule(release_track=self.release_track)

    def ParseExportPolicy(self, volume, export_policy):
        """Parses Export Policy for Volume into a config.

    Args:
      volume: The Cloud NetApp Volume message object
      export_policy: the Export Policy message object.

    Returns:
      Volume message populated with Export Policy values.

    """
        if not export_policy:
            return
        export_policy_config = self.messages.ExportPolicy()
        for policy in export_policy:
            simple_export_policy_rule = self.messages.SimpleExportPolicyRule()
            for key, val in policy.items():
                if key == 'allowed-clients':
                    simple_export_policy_rule.allowedClients = val
                if key == 'access-type':
                    simple_export_policy_rule.accessType = self.messages.SimpleExportPolicyRule.AccessTypeValueValuesEnum.lookup_by_name(val)
                if key == 'has-root-access':
                    simple_export_policy_rule.hasRootAccess = val
                if key == 'kerberos-5-read-only':
                    simple_export_policy_rule.kerberos5ReadOnly = val
                if key == 'kerberos-5-read-write':
                    simple_export_policy_rule.kerberos5ReadWrite = val
                if key == 'kerberos-5i-read-only':
                    simple_export_policy_rule.kerberos5iReadOnly = val
                if key == 'kerberos-5i-read-write':
                    simple_export_policy_rule.kerberos5iReadWrite = val
                if key == 'kerberos-5p-read-only':
                    simple_export_policy_rule.kerberos5pReadOnly = val
                if key == 'kerberos-5p-read-write':
                    simple_export_policy_rule.kerberos5pReadWrite = val
                if key == 'nfsv3':
                    simple_export_policy_rule.nfsv3 = val
                if key == 'nfsv4':
                    simple_export_policy_rule.nfsv4 = val
            export_policy_config.rules.append(simple_export_policy_rule)
        volume.exportPolicy = export_policy_config

    def ParseProtocols(self, volume, protocols):
        """Parses Protocols from a list of Protocol Enums into the given volume.

    Args:
      volume: The Cloud NetApp Volume message object
      protocols: A list of protocol enums

    Returns:
      Volume message populated with protocol values.

    """
        protocols_config = []
        for protocol in protocols:
            protocols_config.append(protocol)
        volume.protocols = protocols_config

    def ParseSnapshotPolicy(self, volume, snapshot_policy):
        """Parses Snapshot Policy from a list of snapshot schedules into a given Volume.

    Args:
      volume: The Cloud NetApp Volume message object
      snapshot_policy: A list of snapshot policies (schedules) to parse

    Returns:
      Volume messages populated with snapshotPolicy field
    """
        if not snapshot_policy:
            return
        volume.snapshotPolicy = self.messages.SnapshotPolicy()
        volume.snapshotPolicy.enabled = True
        for name, snapshot_schedule in snapshot_policy.items():
            if name == 'hourly_snapshot':
                schedule = self.messages.HourlySchedule()
                schedule.snapshotsToKeep = snapshot_schedule.get('snapshots-to-keep')
                schedule.minute = snapshot_schedule.get('minute', 0)
                volume.snapshotPolicy.hourlySchedule = schedule
            elif name == 'daily_snapshot':
                schedule = self.messages.DailySchedule()
                schedule.snapshotsToKeep = snapshot_schedule.get('snapshots-to-keep')
                schedule.minute = snapshot_schedule.get('minute', 0)
                schedule.hour = snapshot_schedule.get('hour', 0)
                volume.snapshotPolicy.dailySchedule = schedule
            elif name == 'weekly_snapshot':
                schedule = self.messages.WeeklySchedule()
                schedule.snapshotsToKeep = snapshot_schedule.get('snapshots-to-keep')
                schedule.minute = snapshot_schedule.get('minute', 0)
                schedule.hour = snapshot_schedule.get('hour', 0)
                schedule.day = snapshot_schedule.get('day', 'Sunday')
                volume.snapshotPolicy.weeklySchedule = schedule
            elif name == 'monthly-snapshot':
                schedule = self.messages.MonthlySchedule()
                schedule.snapshotsToKeep = snapshot_schedule.get('snapshots-to-keep')
                schedule.minute = snapshot_schedule.get('minute', 0)
                schedule.hour = snapshot_schedule.get('hour', 0)
                schedule.day = snapshot_schedule.get('day', 1)
                volume.snapshotPolicy.monthlySchedule = schedule

    def UpdateVolume(self, volume_ref, volume_config, update_mask):
        """Send a Patch request for the Cloud NetApp Volume."""
        update_request = self.messages.NetappProjectsLocationsVolumesPatchRequest(volume=volume_config, name=volume_ref.RelativeName(), updateMask=update_mask)
        update_op = self.client.projects_locations_volumes.Patch(update_request)
        return update_op

    def ParseVolumeConfig(self, name=None, capacity=None, description=None, storage_pool=None, protocols=None, share_name=None, export_policy=None, unix_permissions=None, smb_settings=None, snapshot_policy=None, snap_reserve=None, snapshot_directory=None, security_style=None, enable_kerberos=None, snapshot=None, backup=None, restricted_actions=None, backup_config=None, large_capacity=None, multiple_endpoints=None, tiering_policy=None, labels=None):
        """Parses the command line arguments for Create Volume into a config.

    Args:
      name: the name of the Volume
      capacity: the storage capacity of the Volume.
      description: the description of the Volume.
      storage_pool: the Storage Pool the Volume is attached to.
      protocols: the type of fileshare protocol of the Volume.
      share_name: the share name or mount point of the Volume.
      export_policy: the export policy of the Volume if NFS.
      unix_permissions: the Unix permissions for the Volume.
      smb_settings: the SMB settings for the Volume.
      snapshot_policy: the Snapshot Policy for the Volume
      snap_reserve: the snap reserve (double) for the Volume
      snapshot_directory: Bool on whether to use snapshot directory for Volume
      security_style: the security style of the Volume
      enable_kerberos: Bool on whether to use kerberos for Volume
      snapshot: the snapshot name to create Volume from
      backup: the backup to create the Volume from.
      restricted_actions: the actions to be restricted on a Volume
      backup_config: the Backup Config attached to the Volume
      large_capacity: Bool on whether to use large capacity for Volume
      multiple_endpoints: Bool on whether to use multiple endpoints for Volume
      tiering_policy: the tiering policy for the volume.
      labels: the parsed labels value.

    Returns:
      the configuration that will be used as the request body for creating a
      Cloud NetApp Files Volume.
    """
        volume = self.messages.Volume()
        volume.name = name
        volume.capacityGib = capacity
        volume.description = description
        volume.labels = labels
        volume.storagePool = storage_pool
        volume.shareName = share_name
        self.ParseExportPolicy(volume, export_policy)
        self.ParseProtocols(volume, protocols)
        volume.unixPermissions = unix_permissions
        volume.smbSettings = smb_settings
        self.ParseSnapshotPolicy(volume, snapshot_policy)
        volume.snapReserve = snap_reserve
        volume.snapshotDirectory = snapshot_directory
        volume.securityStyle = security_style
        volume.kerberosEnabled = enable_kerberos
        restore_parameters = self.messages.RestoreParameters()
        if snapshot is not None:
            restore_parameters.sourceSnapshot = snapshot
        if backup is not None:
            restore_parameters.sourceBackup = backup
        if backup is None and snapshot is None:
            restore_parameters = None
        volume.restoreParameters = restore_parameters
        volume.restrictedActions = restricted_actions
        if backup_config is not None:
            self.ParseBackupConfig(volume, backup_config)
        if large_capacity is not None:
            volume.largeCapacity = large_capacity
        if multiple_endpoints is not None:
            volume.multipleEndpoints = multiple_endpoints
        if tiering_policy is not None:
            self.ParseTieringPolicy(volume, tiering_policy)
        return volume

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

    def ParseBackupConfig(self, volume, backup_config):
        """Parses Backup Config for Volume into a config.

    Args:
      volume: The Cloud NetApp Volume message object.
      backup_config: the Backup Config message object.

    Returns:
      Volume message populated with Backup Config values.

    """
        backup_config_message = self.messages.BackupConfig()
        for backup_policy in backup_config.get('backup-policies', []):
            backup_config_message.backupPolicies.append(backup_policy)
        backup_config_message.backupVault = backup_config.get('backup-vault', '')
        backup_config_message.scheduledBackupEnabled = backup_config.get('enable-scheduled-backups', None)
        volume.backupConfig = backup_config_message

    def ParseRestoreParameters(self, volume, snapshot, backup):
        """Parses Restore Parameters for Volume into a config."""
        restore_parameters = self.messages.RestoreParameters()
        if snapshot:
            restore_parameters.sourceSnapshot = snapshot
        if backup:
            restore_parameters.sourceBackup = backup
        volume.restoreParameters = restore_parameters

    def ParseTieringPolicy(self, volume, tiering_policy):
        """Parses Tiering Policy for Volume into a config.

    Args:
      volume: The Cloud NetApp Volume message object.
      tiering_policy: the tiering policy message object.

    Returns:
      Volume message populated with Tiering Policy values.
    """
        tiering_policy_message = self.messages.TieringPolicy()
        tiering_policy_message.tierAction = tiering_policy.get('tier-action')
        tiering_policy_message.coolingThresholdDays = tiering_policy.get('cooling-threshold-days')
        volume.tieringPolicy = tiering_policy_message