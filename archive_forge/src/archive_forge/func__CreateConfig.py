from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_ALPHA
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_BETA
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_GA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_ALPHA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_BETA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_GA
def _CreateConfig(messages, flags, is_composer_v1):
    """Creates environment config from parameters, returns None if config is empty."""
    node_config = _CreateNodeConfig(messages, flags)
    if not (node_config or flags.node_count or flags.kms_key or flags.image_version or flags.env_variables or flags.airflow_config_overrides or flags.python_version or flags.airflow_executor_type or flags.maintenance_window_start or flags.maintenance_window_end or flags.maintenance_window_recurrence or flags.private_environment or flags.web_server_access_control or flags.cloud_sql_machine_type or flags.web_server_machine_type or flags.scheduler_cpu or flags.worker_cpu or flags.web_server_cpu or flags.scheduler_memory_gb or flags.worker_memory_gb or flags.web_server_memory_gb or flags.scheduler_storage_gb or flags.worker_storage_gb or flags.web_server_storage_gb or flags.environment_size or flags.min_workers or flags.max_workers or flags.scheduler_count or flags.airflow_database_retention_days or flags.triggerer_cpu or flags.triggerer_memory or flags.enable_triggerer or flags.enable_scheduled_snapshot_creation or flags.snapshot_creation_schedule or flags.snapshot_location or flags.snapshot_schedule_timezone or flags.enable_cloud_data_lineage_integration or flags.disable_cloud_data_lineage_integration):
        return None
    config = messages.EnvironmentConfig()
    if flags.node_count:
        config.nodeCount = flags.node_count
    if node_config:
        config.nodeConfig = node_config
    if flags.kms_key:
        config.encryptionConfig = messages.EncryptionConfig(kmsKeyName=flags.kms_key)
    if flags.environment_size:
        if flags.release_track == base.ReleaseTrack.GA:
            config.environmentSize = ENVIRONMENT_SIZE_GA.GetEnumForChoice(flags.environment_size)
        elif flags.release_track == base.ReleaseTrack.BETA:
            config.environmentSize = ENVIRONMENT_SIZE_BETA.GetEnumForChoice(flags.environment_size)
        elif flags.release_track == base.ReleaseTrack.ALPHA:
            config.environmentSize = ENVIRONMENT_SIZE_ALPHA.GetEnumForChoice(flags.environment_size)
    if flags.image_version or flags.env_variables or flags.airflow_config_overrides or flags.python_version or flags.airflow_executor_type or (flags.scheduler_count and is_composer_v1) or flags.enable_cloud_data_lineage_integration or flags.disable_cloud_data_lineage_integration:
        config.softwareConfig = messages.SoftwareConfig()
        if flags.image_version:
            config.softwareConfig.imageVersion = flags.image_version
        if flags.env_variables:
            config.softwareConfig.envVariables = api_util.DictToMessage(flags.env_variables, messages.SoftwareConfig.EnvVariablesValue)
        if flags.airflow_config_overrides:
            config.softwareConfig.airflowConfigOverrides = api_util.DictToMessage(flags.airflow_config_overrides, messages.SoftwareConfig.AirflowConfigOverridesValue)
        if flags.python_version:
            config.softwareConfig.pythonVersion = flags.python_version
        if flags.airflow_executor_type:
            config.softwareConfig.airflowExecutorType = ConvertToTypeEnum(messages.SoftwareConfig.AirflowExecutorTypeValueValuesEnum, flags.airflow_executor_type)
        if flags.support_web_server_plugins is not None:
            if flags.support_web_server_plugins:
                config.softwareConfig.webServerPluginsMode = messages.SoftwareConfig.WebServerPluginsModeValueValuesEnum.PLUGINS_ENABLED
            else:
                config.softwareConfig.webServerPluginsMode = messages.SoftwareConfig.WebServerPluginsModeValueValuesEnum.PLUGINS_DISABLED
        if flags.scheduler_count and is_composer_v1:
            config.softwareConfig.schedulerCount = flags.scheduler_count
        if flags.enable_cloud_data_lineage_integration or flags.disable_cloud_data_lineage_integration:
            config.softwareConfig.cloudDataLineageIntegration = messages.CloudDataLineageIntegration(enabled=True if flags.enable_cloud_data_lineage_integration else False)
    if flags.maintenance_window_start:
        assert flags.maintenance_window_end, 'maintenance_window_end is missing'
        assert flags.maintenance_window_recurrence, 'maintenance_window_recurrence is missing'
        config.maintenanceWindow = messages.MaintenanceWindow(startTime=flags.maintenance_window_start.isoformat(), endTime=flags.maintenance_window_end.isoformat(), recurrence=flags.maintenance_window_recurrence)
    if flags.airflow_database_retention_days is not None:
        if flags.airflow_database_retention_days == 0:
            config.dataRetentionConfig = messages.DataRetentionConfig(airflowMetadataRetentionConfig=messages.AirflowMetadataRetentionPolicyConfig(retentionMode=messages.AirflowMetadataRetentionPolicyConfig.RetentionModeValueValuesEnum.RETENTION_MODE_DISABLED))
        else:
            config.dataRetentionConfig = messages.DataRetentionConfig(airflowMetadataRetentionConfig=messages.AirflowMetadataRetentionPolicyConfig(retentionDays=flags.airflow_database_retention_days, retentionMode=messages.AirflowMetadataRetentionPolicyConfig.RetentionModeValueValuesEnum.RETENTION_MODE_ENABLED))
    if flags.enable_scheduled_snapshot_creation:
        config.recoveryConfig = messages.RecoveryConfig(scheduledSnapshotsConfig=messages.ScheduledSnapshotsConfig(enabled=flags.enable_scheduled_snapshot_creation, snapshotCreationSchedule=flags.snapshot_creation_schedule, snapshotLocation=flags.snapshot_location, timeZone=flags.snapshot_schedule_timezone))
    if flags.private_environment or flags.enable_private_builds_only or flags.disable_private_builds_only:
        private_cluster_config = None
        networking_config = None
        if flags.private_endpoint or flags.master_ipv4_cidr:
            private_cluster_config = messages.PrivateClusterConfig(enablePrivateEndpoint=flags.private_endpoint, masterIpv4CidrBlock=flags.master_ipv4_cidr)
        if flags.connection_type:
            if flags.release_track == base.ReleaseTrack.GA:
                connection_type = CONNECTION_TYPE_FLAG_GA.GetEnumForChoice(flags.connection_type)
            elif flags.release_track == base.ReleaseTrack.BETA:
                connection_type = CONNECTION_TYPE_FLAG_BETA.GetEnumForChoice(flags.connection_type)
            elif flags.release_track == base.ReleaseTrack.ALPHA:
                connection_type = CONNECTION_TYPE_FLAG_ALPHA.GetEnumForChoice(flags.connection_type)
            networking_config = messages.NetworkingConfig(connectionType=connection_type)
        private_env_config_args = {'enablePrivateEnvironment': flags.private_environment, 'privateClusterConfig': private_cluster_config, 'networkingConfig': networking_config}
        if flags.web_server_ipv4_cidr is not None:
            private_env_config_args['webServerIpv4CidrBlock'] = flags.web_server_ipv4_cidr
        if flags.cloud_sql_ipv4_cidr is not None:
            private_env_config_args['cloudSqlIpv4CidrBlock'] = flags.cloud_sql_ipv4_cidr
        if flags.composer_network_ipv4_cidr is not None:
            private_env_config_args['cloudComposerNetworkIpv4CidrBlock'] = flags.composer_network_ipv4_cidr
        if flags.privately_used_public_ips is not None:
            private_env_config_args['enablePrivatelyUsedPublicIps'] = flags.privately_used_public_ips
        if flags.connection_subnetwork is not None:
            private_env_config_args['cloudComposerConnectionSubnetwork'] = flags.connection_subnetwork
        if flags.enable_private_builds_only or flags.disable_private_builds_only:
            private_env_config_args['enablePrivateBuildsOnly'] = True if flags.enable_private_builds_only else False
        config.privateEnvironmentConfig = messages.PrivateEnvironmentConfig(**private_env_config_args)
    if flags.web_server_access_control is not None:
        config.webServerNetworkAccessControl = BuildWebServerNetworkAccessControl(flags.web_server_access_control, flags.release_track)
    if flags.enable_high_resilience:
        config.resilienceMode = messages.EnvironmentConfig.ResilienceModeValueValuesEnum.HIGH_RESILIENCE
    if flags.enable_logs_in_cloud_logging_only:
        task_logs_retention_config = messages.TaskLogsRetentionConfig(storageMode=messages.TaskLogsRetentionConfig.StorageModeValueValuesEnum.CLOUD_LOGGING_ONLY)
        config.dataRetentionConfig = messages.DataRetentionConfig(taskLogsRetentionConfig=task_logs_retention_config)
    if flags.disable_logs_in_cloud_logging_only:
        task_logs_retention_config = messages.TaskLogsRetentionConfig(storageMode=messages.TaskLogsRetentionConfig.StorageModeValueValuesEnum.CLOUD_LOGGING_AND_CLOUD_STORAGE)
        config.dataRetentionConfig = messages.DataRetentionConfig(taskLogsRetentionConfig=task_logs_retention_config)
    if flags.cloud_sql_machine_type:
        config.databaseConfig = messages.DatabaseConfig(machineType=flags.cloud_sql_machine_type)
    if flags.cloud_sql_preferred_zone:
        config.databaseConfig = messages.DatabaseConfig(zone=flags.cloud_sql_preferred_zone)
    if flags.web_server_machine_type:
        config.webServerConfig = messages.WebServerConfig(machineType=flags.web_server_machine_type)
    if flags.enable_master_authorized_networks:
        networks = flags.master_authorized_networks if flags.master_authorized_networks else []
        config.masterAuthorizedNetworksConfig = messages.MasterAuthorizedNetworksConfig(enabled=True, cidrBlocks=[messages.CidrBlock(cidrBlock=network) for network in networks])
    composer_v2_flags = [flags.scheduler_cpu, flags.worker_cpu, flags.web_server_cpu, flags.scheduler_memory_gb, flags.worker_memory_gb, flags.web_server_memory_gb, flags.scheduler_storage_gb, flags.worker_storage_gb, flags.web_server_storage_gb, flags.min_workers, flags.max_workers, flags.triggerer_memory_gb, flags.triggerer_cpu, flags.enable_triggerer, flags.triggerer_count, flags.dag_processor_cpu, flags.dag_processor_count, flags.dag_processor_memory_gb, flags.dag_processor_storage_gb]
    composer_v2_flag_used = any((flag is not None for flag in composer_v2_flags))
    if composer_v2_flag_used or (flags.scheduler_count and (not is_composer_v1)):
        config.workloadsConfig = _CreateWorkloadConfig(messages, flags)
    return config