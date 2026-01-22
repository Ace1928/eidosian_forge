from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _ConstructClusterAndMaskForPatchRequestBeta(alloydb_messages, args):
    """Returns the cluster resource for patch request."""
    cluster, update_masks = _ConstructClusterAndMaskForPatchRequestGA(alloydb_messages, args)
    if args.automated_backup_enforced_retention is not None:
        if cluster.automatedBackupPolicy is None:
            cluster.automatedBackupPolicy = _ConstructAutomatedBackupPolicy(alloydb_messages, args)
        update_masks.append('automated_backup_policy.enforced_retention')
        cluster.automatedBackupPolicy = _AddEnforcedRetentionToAutomatedBackupPolicy(cluster.automatedBackupPolicy, args)
    if args.continuous_backup_enforced_retention is not None:
        if cluster.continuousBackupConfig is None:
            cluster.continuousBackupConfig = _ConstructContinuousBackupConfig(alloydb_messages, args)
        update_masks.append('continuous_backup_config.enforced_retention')
        cluster.continuousBackupConfig = _AddEnforcedRetentionToContinuousBackupConfig(cluster.continuousBackupConfig, args)
    update_maintenance_window = args.maintenance_window_any or args.maintenance_window_day or args.maintenance_window_hour
    update_deny_period = args.remove_deny_maintenance_period or args.deny_maintenance_period_start_date or args.deny_maintenance_period_end_date or args.deny_maintenance_period_time
    if update_maintenance_window or update_deny_period:
        cluster.maintenanceUpdatePolicy = alloydb_messages.MaintenanceUpdatePolicy()
        if update_maintenance_window:
            cluster.maintenanceUpdatePolicy.maintenanceWindows = _ConstructMaintenanceWindows(alloydb_messages, args, update=True)
            update_masks.append('maintenance_update_policy.maintenance_windows')
        if update_deny_period:
            cluster.maintenanceUpdatePolicy.denyMaintenancePeriods = _ConstructDenyPeriods(alloydb_messages, args, update=True)
            update_masks.append('maintenance_update_policy.deny_maintenance_periods')
    return (cluster, update_masks)