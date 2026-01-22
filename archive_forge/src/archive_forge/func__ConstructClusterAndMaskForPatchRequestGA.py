from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _ConstructClusterAndMaskForPatchRequestGA(alloydb_messages, args):
    """Returns the cluster resource for patch request."""
    cluster = alloydb_messages.Cluster()
    update_masks = []
    continuous_backup_update_masks = []
    if args.disable_automated_backup or args.automated_backup_days_of_week or args.clear_automated_backup:
        cluster.automatedBackupPolicy = _ConstructAutomatedBackupPolicy(alloydb_messages, args)
        update_masks.append('automated_backup_policy')
    if args.enable_continuous_backup:
        continuous_backup_update_masks.append('continuous_backup_config.enabled')
    elif args.enable_continuous_backup is False:
        update_masks.append('continuous_backup_config')
        cluster.continuousBackupConfig = _ConstructContinuousBackupConfig(alloydb_messages, args, update=True)
        return (cluster, update_masks)
    if args.continuous_backup_recovery_window_days:
        continuous_backup_update_masks.append('continuous_backup_config.recovery_window_days')
    if args.continuous_backup_encryption_key or args.clear_continuous_backup_encryption_key:
        continuous_backup_update_masks.append('continuous_backup_config.encryption_config')
    update_masks.extend(continuous_backup_update_masks)
    if continuous_backup_update_masks:
        cluster.continuousBackupConfig = _ConstructContinuousBackupConfig(alloydb_messages, args, update=True)
    return (cluster, update_masks)