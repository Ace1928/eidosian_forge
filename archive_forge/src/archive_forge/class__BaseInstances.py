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
class _BaseInstances(object):
    """Common utility functions for sql instance commands."""

    @classmethod
    def _ConstructBaseSettingsFromArgs(cls, sql_messages, args, instance=None, release_track=DEFAULT_RELEASE_TRACK):
        """Constructs instance settings from the command line arguments.

    Args:
      sql_messages: module, The messages module that should be used.
      args: argparse.Namespace, The arguments that this command was invoked
        with.
      instance: sql_messages.DatabaseInstance, The original instance, for
        settings that depend on the previous state.
      release_track: base.ReleaseTrack, the release track that this was run
        under.

    Returns:
      A settings object representing the instance settings.

    Raises:
      ToolException: An error other than http error occurred while executing the
          command.
    """
        if 'authorized_gae_apps' not in args:
            args.authorized_gae_apps = None
        if 'follow_gae_app' not in args:
            args.follow_gae_app = None
        if 'pricing_plan' not in args:
            args.pricing_plan = 'PER_USE'
        settings = sql_messages.Settings(kind='sql#settings', tier=reducers.MachineType(instance, args.tier, args.memory, args.cpu), pricingPlan=_ParsePricingPlan(sql_messages, args.pricing_plan), replicationType=_ParseReplicationType(sql_messages, args.replication), activationPolicy=_ParseActivationPolicy(sql_messages, args.activation_policy))
        if args.authorized_gae_apps:
            settings.authorizedGaeApplications = args.authorized_gae_apps
        if any([args.assign_ip is not None, args.require_ssl is not None, args.authorized_networks]):
            settings.ipConfiguration = sql_messages.IpConfiguration()
            if args.assign_ip is not None:
                cls.SetIpConfigurationEnabled(settings, args.assign_ip)
            if args.authorized_networks:
                cls.SetAuthorizedNetworks(settings, args.authorized_networks, sql_messages.AclEntry)
            if args.require_ssl is not None:
                settings.ipConfiguration.requireSsl = args.require_ssl
        if any([args.follow_gae_app, _GetZone(args), _GetSecondaryZone(args)]):
            settings.locationPreference = sql_messages.LocationPreference(kind='sql#locationPreference', followGaeApplication=args.follow_gae_app, zone=_GetZone(args), secondaryZone=_GetSecondaryZone(args))
        if args.storage_size:
            settings.dataDiskSizeGb = int(args.storage_size / constants.BYTES_TO_GB)
        if args.storage_auto_increase is not None:
            settings.storageAutoResize = args.storage_auto_increase
        if args.IsSpecified('availability_type'):
            settings.availabilityType = _ParseAvailabilityType(sql_messages, args.availability_type)
        if args.IsSpecified('network'):
            if not settings.ipConfiguration:
                settings.ipConfiguration = sql_messages.IpConfiguration()
            settings.ipConfiguration.privateNetwork = reducers.PrivateNetworkUrl(args.network)
        if args.IsKnownAndSpecified('enable_google_private_path'):
            if not settings.ipConfiguration:
                settings.ipConfiguration = sql_messages.IpConfiguration()
            settings.ipConfiguration.enablePrivatePathForGoogleCloudServices = args.enable_google_private_path
        if args.IsKnownAndSpecified('enable_private_service_connect'):
            if not settings.ipConfiguration:
                settings.ipConfiguration = sql_messages.IpConfiguration()
            if not settings.ipConfiguration.pscConfig:
                settings.ipConfiguration.pscConfig = sql_messages.PscConfig()
            settings.ipConfiguration.pscConfig.pscEnabled = args.enable_private_service_connect
        if args.IsSpecified('allowed_psc_projects'):
            if not settings.ipConfiguration:
                settings.ipConfiguration = sql_messages.IpConfiguration()
            if not settings.ipConfiguration.pscConfig:
                settings.ipConfiguration.pscConfig = sql_messages.PscConfig()
            settings.ipConfiguration.pscConfig.allowedConsumerProjects = args.allowed_psc_projects
        if args.IsKnownAndSpecified('clear_allowed_psc_projects'):
            if not settings.ipConfiguration:
                settings.ipConfiguration = sql_messages.IpConfiguration()
            if not settings.ipConfiguration.pscConfig:
                settings.ipConfiguration.pscConfig = sql_messages.PscConfig()
            settings.ipConfiguration.pscConfig.allowedConsumerProjects = []
        if args.deletion_protection is not None:
            settings.deletionProtectionEnabled = args.deletion_protection
        if args.IsSpecified('connector_enforcement'):
            settings.connectorEnforcement = _ParseConnectorEnforcement(sql_messages, args.connector_enforcement)
        if args.recreate_replicas_on_primary_crash is not None:
            settings.recreateReplicasOnPrimaryCrash = args.recreate_replicas_on_primary_crash
        if args.IsSpecified('edition'):
            settings.edition = _ParseEdition(sql_messages, args.edition)
        if args.IsKnownAndSpecified('enable_data_cache'):
            if not settings.dataCacheConfig:
                settings.dataCacheConfig = sql_messages.DataCacheConfig()
            settings.dataCacheConfig.dataCacheEnabled = args.enable_data_cache
        if args.IsSpecified('ssl_mode'):
            if not settings.ipConfiguration:
                settings.ipConfiguration = sql_messages.IpConfiguration()
            settings.ipConfiguration.sslMode = _ParseSslMode(sql_messages, args.ssl_mode)
        if args.enable_google_ml_integration is not None:
            settings.enableGoogleMlIntegration = args.enable_google_ml_integration
        if IsBetaOrNewer(release_track):
            if args.IsSpecified('storage_auto_increase_limit'):
                if instance and instance.settings.storageAutoResize or args.storage_auto_increase:
                    settings.storageAutoResizeLimit = args.storage_auto_increase_limit or 0
                else:
                    raise exceptions.RequiredArgumentException('--storage-auto-increase', 'To set the storage capacity limit using [--storage-auto-increase-limit], [--storage-auto-increase] must be enabled.')
            if args.replication_lag_max_seconds_for_recreate is not None:
                settings.replicationLagMaxSeconds = args.replication_lag_max_seconds_for_recreate
        return settings

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

    @classmethod
    def _ConstructPatchSettingsFromArgs(cls, sql_messages, args, instance, release_track=DEFAULT_RELEASE_TRACK):
        """Constructs patch settings object from base settings and args."""
        original_settings = instance.settings
        settings = cls._ConstructBaseSettingsFromArgs(sql_messages, args, instance, release_track)
        if args.clear_gae_apps:
            settings.authorizedGaeApplications = []
        if any([args.follow_gae_app, _GetZone(args), _GetSecondaryZone(args)]):
            settings.locationPreference = sql_messages.LocationPreference(kind='sql#locationPreference', followGaeApplication=args.follow_gae_app, zone=_GetZone(args), secondaryZone=_GetSecondaryZone(args))
        if args.clear_authorized_networks:
            if not settings.ipConfiguration:
                settings.ipConfiguration = sql_messages.IpConfiguration()
            settings.ipConfiguration.authorizedNetworks = []
        if args.enable_database_replication is not None:
            settings.databaseReplicationEnabled = args.enable_database_replication
        backup_configuration = reducers.BackupConfiguration(sql_messages, instance, backup_enabled=not args.no_backup, backup_location=args.backup_location, backup_start_time=args.backup_start_time, enable_bin_log=args.enable_bin_log, enable_point_in_time_recovery=args.enable_point_in_time_recovery, retained_backups_count=args.retained_backups_count, retained_transaction_log_days=args.retained_transaction_log_days)
        if backup_configuration:
            cls.AddBackupConfigToSettings(settings, backup_configuration)
        settings.databaseFlags = reducers.DatabaseFlags(sql_messages, original_settings, database_flags=args.database_flags, clear_database_flags=args.clear_database_flags)
        settings.maintenanceWindow = reducers.MaintenanceWindow(sql_messages, instance, maintenance_release_channel=args.maintenance_release_channel, maintenance_window_day=args.maintenance_window_day, maintenance_window_hour=args.maintenance_window_hour)
        if args.remove_deny_maintenance_period:
            settings.denyMaintenancePeriods = []
        if args.deny_maintenance_period_start_date or args.deny_maintenance_period_end_date or args.deny_maintenance_period_time:
            settings.denyMaintenancePeriods = []
            settings.denyMaintenancePeriods.append(reducers.DenyMaintenancePeriod(sql_messages, instance, deny_maintenance_period_start_date=args.deny_maintenance_period_start_date, deny_maintenance_period_end_date=args.deny_maintenance_period_end_date, deny_maintenance_period_time=args.deny_maintenance_period_time))
        settings.insightsConfig = reducers.InsightsConfig(sql_messages, insights_config_query_insights_enabled=args.insights_config_query_insights_enabled, insights_config_query_string_length=args.insights_config_query_string_length, insights_config_record_application_tags=args.insights_config_record_application_tags, insights_config_record_client_address=args.insights_config_record_client_address, insights_config_query_plans_per_minute=args.insights_config_query_plans_per_minute)
        if args.active_directory_domain is not None:
            settings.activeDirectoryConfig = reducers.ActiveDirectoryConfig(sql_messages, args.active_directory_domain)
        settings.passwordValidationPolicy = reducers.PasswordPolicy(sql_messages, password_policy_min_length=args.password_policy_min_length, password_policy_complexity=args.password_policy_complexity, password_policy_reuse_interval=args.password_policy_reuse_interval, password_policy_disallow_username_substring=args.password_policy_disallow_username_substring, password_policy_password_change_interval=args.password_policy_password_change_interval, enable_password_policy=args.enable_password_policy)
        settings.sqlServerAuditConfig = reducers.SqlServerAuditConfig(sql_messages, bucket=args.audit_bucket_path, retention_interval=args.audit_retention_interval, upload_interval=args.audit_upload_interval)
        if args.threads_per_core is not None:
            settings.advancedMachineFeatures = sql_messages.AdvancedMachineFeatures()
            settings.advancedMachineFeatures.threadsPerCore = args.threads_per_core
        if IsBetaOrNewer(release_track):
            labels_diff = labels_util.ExplicitNullificationDiff.FromUpdateArgs(args)
            labels_update = labels_diff.Apply(sql_messages.Settings.UserLabelsValue, instance.settings.userLabels)
            if labels_update.needs_update:
                settings.userLabels = labels_update.labels
            if args.allocated_ip_range_name:
                if not settings.ipConfiguration:
                    settings.ipConfiguration = sql_messages.IpConfiguration()
                settings.ipConfiguration.allocatedIpRange = args.allocated_ip_range_name
        if _IsAlpha(release_track):
            pass
        return settings

    @classmethod
    def _ConstructBaseInstanceFromArgs(cls, sql_messages, args, original=None, instance_ref=None, release_track=DEFAULT_RELEASE_TRACK):
        """Construct a Cloud SQL instance from command line args.

    Args:
      sql_messages: module, The messages module that should be used.
      args: argparse.Namespace, The CLI arg namespace.
      original: sql_messages.DatabaseInstance, The original instance, if some of
        it might be used to fill fields in the new one.
      instance_ref: reference to DatabaseInstance object, used to fill project
        and instance information.
      release_track: base.ReleaseTrack, the release track that this was run
        under.

    Returns:
      sql_messages.DatabaseInstance, The constructed (and possibly partial)
      database instance.

    Raises:
      ToolException: An error other than http error occurred while executing the
          command.
    """
        del args, original, release_track
        instance_resource = sql_messages.DatabaseInstance(kind='sql#instance')
        if instance_ref:
            cls.SetProjectAndInstanceFromRef(instance_resource, instance_ref)
        return instance_resource

    @classmethod
    def ConstructCreateInstanceFromArgs(cls, sql_messages, args, original=None, instance_ref=None, release_track=DEFAULT_RELEASE_TRACK):
        """Constructs Instance for create request from base instance and args."""
        ShowZoneDeprecationWarnings(args)
        instance_resource = cls._ConstructBaseInstanceFromArgs(sql_messages, args, original, instance_ref)
        instance_resource.region = reducers.Region(args.region, _GetZone(args), _GetSecondaryZone(args))
        instance_resource.databaseVersion = ParseDatabaseVersion(sql_messages, args.database_version)
        instance_resource.masterInstanceName = args.master_instance_name
        instance_resource.rootPassword = args.root_password
        if IsBetaOrNewer(release_track) and args.IsSpecified('source_ip_address'):
            on_premises_configuration = reducers.OnPremisesConfiguration(sql_messages, args.source_ip_address, args.source_port)
            instance_resource.onPremisesConfiguration = on_premises_configuration
            return instance_resource
        instance_resource.settings = cls._ConstructCreateSettingsFromArgs(sql_messages, args, original, release_track)
        if args.master_instance_name:
            replication = sql_messages.Settings.ReplicationTypeValueValuesEnum.ASYNCHRONOUS
            if args.replica_type == 'FAILOVER':
                instance_resource.replicaConfiguration = sql_messages.ReplicaConfiguration(kind='sql#demoteMasterMysqlReplicaConfiguration', failoverTarget=True)
            if args.cascadable_replica:
                if instance_resource.replicaConfiguration:
                    instance_resource.replicaConfiguration.cascadableReplica = args.cascadable_replica
                else:
                    instance_resource.replicaConfiguration = sql_messages.ReplicaConfiguration(kind='sql#replicaConfiguration', cascadableReplica=args.cascadable_replica)
        else:
            replication = sql_messages.Settings.ReplicationTypeValueValuesEnum.SYNCHRONOUS
        if not args.replication:
            instance_resource.settings.replicationType = replication
        if args.failover_replica_name:
            instance_resource.failoverReplica = sql_messages.DatabaseInstance.FailoverReplicaValue(name=args.failover_replica_name)
        if args.collation:
            instance_resource.settings.collation = args.collation
        if IsBetaOrNewer(release_track) and args.IsSpecified('master_username'):
            if not args.IsSpecified('master_instance_name'):
                raise exceptions.RequiredArgumentException('--master-instance-name', 'To create a read replica of an external master instance, [--master-instance-name] must be specified')
            if not (args.IsSpecified('master_password') or args.IsSpecified('prompt_for_master_password')):
                raise exceptions.RequiredArgumentException('--master-password', 'To create a read replica of an external master instance, [--master-password] or [--prompt-for-master-password] must be specified')
            if args.prompt_for_master_password:
                args.master_password = console_io.PromptPassword('Master Instance Password: ')
            instance_resource.replicaConfiguration = reducers.ReplicaConfiguration(sql_messages, args.master_username, args.master_password, args.master_dump_file_path, args.master_ca_certificate_path, args.client_certificate_path, args.client_key_path)
        is_primary = instance_resource.masterInstanceName is None
        key_name = _GetAndValidateCmekKeyName(args, is_primary)
        if key_name:
            config = sql_messages.DiskEncryptionConfiguration(kind='sql#diskEncryptionConfiguration', kmsKeyName=key_name)
            instance_resource.diskEncryptionConfiguration = config
        return instance_resource

    @classmethod
    def ConstructPatchInstanceFromArgs(cls, sql_messages, args, original, instance_ref=None, release_track=DEFAULT_RELEASE_TRACK):
        """Constructs Instance for patch request from base instance and args."""
        instance_resource = cls._ConstructBaseInstanceFromArgs(sql_messages, args, original, instance_ref)
        instance_resource.databaseVersion = ParseDatabaseVersion(sql_messages, args.database_version)
        instance_resource.maintenanceVersion = args.maintenance_version
        instance_resource.settings = cls._ConstructPatchSettingsFromArgs(sql_messages, args, original, release_track)
        if args.upgrade_sql_network_architecture:
            instance_resource.sqlNetworkArchitecture = sql_messages.DatabaseInstance.SqlNetworkArchitectureValueValuesEnum.NEW_NETWORK_ARCHITECTURE
        if args.IsSpecified('simulate_maintenance_event'):
            instance_resource.maintenanceVersion = original.maintenanceVersion
            api_util.InstancesV1Beta4.PrintAndConfirmSimulatedMaintenanceEvent()
        if args.IsSpecified('maintenance_version') and args.maintenance_version == original.maintenanceVersion:
            api_util.InstancesV1Beta4.PrintAndConfirmSimulatedMaintenanceEvent()
        if IsBetaOrNewer(release_track):
            if args.IsKnownAndSpecified('failover_dr_replica_name'):
                replication_cluster = sql_messages.ReplicationCluster()
                replication_cluster.failoverDrReplicaName = args.failover_dr_replica_name
                instance_resource.replicationCluster = replication_cluster
            if args.IsKnownAndSpecified('clear_failover_dr_replica_name'):
                if instance_resource.replicationCluster is not None:
                    instance_resource.replicationCluster.ClearFailoverDrReplicaName()
        return instance_resource