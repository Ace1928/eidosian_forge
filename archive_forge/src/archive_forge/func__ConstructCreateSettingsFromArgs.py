from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import instance_prop_reducers as reducers
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
@classmethod
def _ConstructCreateSettingsFromArgs(cls, sql_messages, args, instance=None, release_track=DEFAULT_RELEASE_TRACK):
    """Constructs create settings object from base settings and args."""
    original_settings = instance.settings if instance else None
    settings = cls._ConstructBaseSettingsFromArgs(sql_messages, args, instance, release_track)
    backup_configuration = reducers.BackupConfiguration(sql_messages, instance, backup_enabled=args.backup, backup_location=args.backup_location, backup_start_time=args.backup_start_time, enable_bin_log=args.enable_bin_log, enable_point_in_time_recovery=args.enable_point_in_time_recovery, retained_backups_count=args.retained_backups_count, retained_transaction_log_days=args.retained_transaction_log_days)
    if backup_configuration:
        cls.AddBackupConfigToSettings(settings, backup_configuration)
    settings.databaseFlags = reducers.DatabaseFlags(sql_messages, original_settings, database_flags=args.database_flags)
    settings.maintenanceWindow = reducers.MaintenanceWindow(sql_messages, instance, maintenance_release_channel=args.maintenance_release_channel, maintenance_window_day=args.maintenance_window_day, maintenance_window_hour=args.maintenance_window_hour)
    if args.deny_maintenance_period_start_date and args.deny_maintenance_period_end_date:
        settings.denyMaintenancePeriods = []
        settings.denyMaintenancePeriods.append(reducers.DenyMaintenancePeriod(sql_messages, instance, deny_maintenance_period_start_date=args.deny_maintenance_period_start_date, deny_maintenance_period_end_date=args.deny_maintenance_period_end_date, deny_maintenance_period_time=args.deny_maintenance_period_time))
    settings.insightsConfig = reducers.InsightsConfig(sql_messages, insights_config_query_insights_enabled=args.insights_config_query_insights_enabled, insights_config_query_string_length=args.insights_config_query_string_length, insights_config_record_application_tags=args.insights_config_record_application_tags, insights_config_record_client_address=args.insights_config_record_client_address, insights_config_query_plans_per_minute=args.insights_config_query_plans_per_minute)
    if args.storage_type:
        settings.dataDiskType = _ParseStorageType(sql_messages, STORAGE_TYPE_PREFIX + args.storage_type)
    if args.active_directory_domain is not None:
        settings.activeDirectoryConfig = reducers.ActiveDirectoryConfig(sql_messages, args.active_directory_domain)
    settings.passwordValidationPolicy = reducers.PasswordPolicy(sql_messages, password_policy_min_length=args.password_policy_min_length, password_policy_complexity=args.password_policy_complexity, password_policy_reuse_interval=args.password_policy_reuse_interval, password_policy_disallow_username_substring=args.password_policy_disallow_username_substring, password_policy_password_change_interval=args.password_policy_password_change_interval, enable_password_policy=args.enable_password_policy)
    settings.sqlServerAuditConfig = reducers.SqlServerAuditConfig(sql_messages, args.audit_bucket_path, args.audit_retention_interval, args.audit_upload_interval)
    if args.time_zone is not None:
        settings.timeZone = args.time_zone
    if args.threads_per_core is not None:
        settings.advancedMachineFeatures = sql_messages.AdvancedMachineFeatures()
        settings.advancedMachineFeatures.threadsPerCore = args.threads_per_core
    if IsBetaOrNewer(release_track):
        settings.userLabels = labels_util.ParseCreateArgs(args, sql_messages.Settings.UserLabelsValue)
        if args.allocated_ip_range_name:
            if not settings.ipConfiguration:
                settings.ipConfiguration = sql_messages.IpConfiguration()
            settings.ipConfiguration.allocatedIpRange = args.allocated_ip_range_name
    if _IsAlpha(release_track):
        pass
    return settings