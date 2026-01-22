from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def BackupConfiguration(sql_messages, instance=None, backup_enabled=None, backup_location=None, backup_start_time=None, enable_bin_log=None, enable_point_in_time_recovery=None, retained_backups_count=None, retained_transaction_log_days=None):
    """Generates the backup configuration for the instance.

  Args:
    sql_messages: module, The messages module that should be used.
    instance: sql_messages.DatabaseInstance, the original instance, if the
      previous state is needed.
    backup_enabled: boolean, True if backup should be enabled.
    backup_location: string, location where to store backups by default.
    backup_start_time: string, start time of backup specified in 24-hour format.
    enable_bin_log: boolean, True if binary logging should be enabled.
    enable_point_in_time_recovery: boolean, True if point-in-time recovery
      (using write-ahead log archiving) should be enabled.
    retained_backups_count: int, how many backups to keep stored.
    retained_transaction_log_days: int, how many days of transaction logs to
      keep stored.

  Returns:
    sql_messages.BackupConfiguration object, or None

  Raises:
    ToolException: Bad combination of arguments.
  """
    should_generate_config = any([backup_location is not None, backup_start_time, enable_bin_log is not None, enable_point_in_time_recovery is not None, retained_backups_count is not None, retained_transaction_log_days is not None, not backup_enabled])
    if not should_generate_config:
        return None
    if not instance or not instance.settings.backupConfiguration:
        backup_config = sql_messages.BackupConfiguration(kind='sql#backupConfiguration', startTime='00:00', enabled=backup_enabled)
    else:
        backup_config = instance.settings.backupConfiguration
    if backup_location is not None:
        backup_config.location = backup_location
        backup_config.enabled = True
    if backup_start_time:
        backup_config.startTime = backup_start_time
        backup_config.enabled = True
    if retained_backups_count is not None:
        backup_retention_settings = backup_config.backupRetentionSettings or sql_messages.BackupRetentionSettings()
        backup_retention_settings.retentionUnit = sql_messages.BackupRetentionSettings.RetentionUnitValueValuesEnum.COUNT
        backup_retention_settings.retainedBackups = retained_backups_count
        backup_config.backupRetentionSettings = backup_retention_settings
        backup_config.enabled = True
    if retained_transaction_log_days is not None:
        backup_config.transactionLogRetentionDays = retained_transaction_log_days
        backup_config.enabled = True
    if not backup_enabled:
        if backup_location is not None or backup_start_time or retained_backups_count is not None or (retained_transaction_log_days is not None):
            raise sql_exceptions.ArgumentError('Argument --no-backup not allowed with --backup-location, --backup-start-time, --retained-backups-count, or --retained-transaction-log-days')
        backup_config.enabled = False
    if enable_bin_log is not None:
        backup_config.binaryLogEnabled = enable_bin_log
    if enable_point_in_time_recovery is not None:
        backup_config.pointInTimeRecoveryEnabled = enable_point_in_time_recovery
        if backup_config.replicationLogArchivingEnabled is not None:
            backup_config.replicationLogArchivingEnabled = enable_point_in_time_recovery
    if retained_transaction_log_days and (not backup_config.binaryLogEnabled) and (not backup_config.pointInTimeRecoveryEnabled):
        raise sql_exceptions.ArgumentError('Argument --retained-transaction-log-days only valid when transaction logs are enabled. To enable transaction logs, use --enable-bin-log for MySQL, and use --enable-point-in-time-recovery for Postgres.')
    return backup_config