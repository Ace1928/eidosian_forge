from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _ConstructAutomatedBackupPolicy(alloydb_messages, args):
    """Returns the automated backup policy based on args."""
    backup_policy = alloydb_messages.AutomatedBackupPolicy()
    if args.disable_automated_backup:
        backup_policy.enabled = False
    elif args.automated_backup_days_of_week:
        backup_policy.enabled = True
        backup_policy.weeklySchedule = alloydb_messages.WeeklySchedule(daysOfWeek=args.automated_backup_days_of_week, startTimes=args.automated_backup_start_times)
        if args.automated_backup_retention_count:
            backup_policy.quantityBasedRetention = alloydb_messages.QuantityBasedRetention(count=args.automated_backup_retention_count)
        elif args.automated_backup_retention_period:
            backup_policy.timeBasedRetention = alloydb_messages.TimeBasedRetention(retentionPeriod='{}s'.format(args.automated_backup_retention_period))
        if args.automated_backup_window:
            backup_policy.backupWindow = '{}s'.format(args.automated_backup_window)
        kms_key = flags.GetAndValidateKmsKeyName(args, flag_overrides=flags.GetAutomatedBackupKmsFlagOverrides())
        if kms_key:
            encryption_config = alloydb_messages.EncryptionConfig()
            encryption_config.kmsKeyName = kms_key
            backup_policy.encryptionConfig = encryption_config
        backup_policy.location = args.region
    return backup_policy