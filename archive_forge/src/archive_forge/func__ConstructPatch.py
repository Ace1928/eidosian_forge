from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import environment_patch_util as patch_util
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import image_versions_util as image_versions_command_util
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
def _ConstructPatch(self, env_ref, args, support_environment_upgrades=False):
    env_obj = environments_api_util.Get(env_ref, release_track=self.ReleaseTrack())
    is_composer_v1 = image_versions_command_util.IsImageVersionStringComposerV1(env_obj.config.softwareConfig.imageVersion)
    params = dict(is_composer_v1=is_composer_v1, env_ref=env_ref, node_count=args.node_count, update_pypi_packages_from_file=args.update_pypi_packages_from_file, clear_pypi_packages=args.clear_pypi_packages, remove_pypi_packages=args.remove_pypi_packages, update_pypi_packages=dict((command_util.SplitRequirementSpecifier(r) for r in args.update_pypi_package)), clear_labels=args.clear_labels, remove_labels=args.remove_labels, update_labels=args.update_labels, clear_airflow_configs=args.clear_airflow_configs, remove_airflow_configs=args.remove_airflow_configs, update_airflow_configs=args.update_airflow_configs, clear_env_variables=args.clear_env_variables, remove_env_variables=args.remove_env_variables, update_env_variables=args.update_env_variables, release_track=self.ReleaseTrack())
    if support_environment_upgrades:
        params['update_image_version'] = self._getImageVersion(args, env_ref, env_obj)
    params['update_web_server_access_control'] = environments_api_util.BuildWebServerAllowedIps(args.update_web_server_allow_ip, args.web_server_allow_all, args.web_server_deny_all)
    if args.cloud_sql_machine_type and (not is_composer_v1):
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V2_ERROR_MSG.format(opt='cloud-sql-machine-type'))
    if args.web_server_machine_type and (not is_composer_v1):
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V2_ERROR_MSG.format(opt='web-server-machine-type'))
    params['cloud_sql_machine_type'] = args.cloud_sql_machine_type
    params['web_server_machine_type'] = args.web_server_machine_type
    if self._support_environment_size:
        if args.environment_size and is_composer_v1:
            raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V1_ERROR_MSG.format(opt='environment-size'))
        if self.ReleaseTrack() == base.ReleaseTrack.GA:
            params['environment_size'] = flags.ENVIRONMENT_SIZE_GA.GetEnumForChoice(args.environment_size)
        elif self.ReleaseTrack() == base.ReleaseTrack.BETA:
            params['environment_size'] = flags.ENVIRONMENT_SIZE_BETA.GetEnumForChoice(args.environment_size)
        elif self.ReleaseTrack() == base.ReleaseTrack.ALPHA:
            params['environment_size'] = flags.ENVIRONMENT_SIZE_ALPHA.GetEnumForChoice(args.environment_size)
    if self._support_autoscaling:
        if args.scheduler_cpu or args.worker_cpu or args.web_server_cpu or args.scheduler_memory or args.worker_memory or args.web_server_memory or args.scheduler_storage or args.worker_storage or args.web_server_storage or args.min_workers or args.max_workers or (args.enable_triggerer or args.disable_triggerer or args.triggerer_count is not None or args.triggerer_cpu or args.triggerer_memory):
            params['workload_updated'] = True
            if is_composer_v1:
                raise command_util.InvalidUserInputError('Workloads Config flags introduced in Composer 2.X cannot be used when updating Composer 1.X environments.')
        if env_obj.config.workloadsConfig:
            if env_obj.config.workloadsConfig.scheduler:
                params['scheduler_cpu'] = env_obj.config.workloadsConfig.scheduler.cpu
                params['scheduler_memory_gb'] = env_obj.config.workloadsConfig.scheduler.memoryGb
                params['scheduler_storage_gb'] = env_obj.config.workloadsConfig.scheduler.storageGb
                params['scheduler_count'] = env_obj.config.workloadsConfig.scheduler.count
            if env_obj.config.workloadsConfig.worker:
                params['worker_cpu'] = env_obj.config.workloadsConfig.worker.cpu
                params['worker_memory_gb'] = env_obj.config.workloadsConfig.worker.memoryGb
                params['worker_storage_gb'] = env_obj.config.workloadsConfig.worker.storageGb
                params['min_workers'] = env_obj.config.workloadsConfig.worker.minCount
                params['max_workers'] = env_obj.config.workloadsConfig.worker.maxCount
            if env_obj.config.workloadsConfig.webServer:
                params['web_server_cpu'] = env_obj.config.workloadsConfig.webServer.cpu
                params['web_server_memory_gb'] = env_obj.config.workloadsConfig.webServer.memoryGb
                params['web_server_storage_gb'] = env_obj.config.workloadsConfig.webServer.storageGb
        if args.scheduler_count is not None:
            params['scheduler_count'] = args.scheduler_count
            if not is_composer_v1:
                params['workload_updated'] = True
        if args.scheduler_cpu is not None:
            params['scheduler_cpu'] = args.scheduler_cpu
        if args.worker_cpu is not None:
            params['worker_cpu'] = args.worker_cpu
        if args.web_server_cpu is not None:
            params['web_server_cpu'] = args.web_server_cpu
        if args.scheduler_memory is not None:
            params['scheduler_memory_gb'] = environments_api_util.MemorySizeBytesToGB(args.scheduler_memory)
        if args.worker_memory is not None:
            params['worker_memory_gb'] = environments_api_util.MemorySizeBytesToGB(args.worker_memory)
        if args.web_server_memory is not None:
            params['web_server_memory_gb'] = environments_api_util.MemorySizeBytesToGB(args.web_server_memory)
        if args.scheduler_storage is not None:
            params['scheduler_storage_gb'] = environments_api_util.MemorySizeBytesToGB(args.scheduler_storage)
        if args.worker_storage is not None:
            params['worker_storage_gb'] = environments_api_util.MemorySizeBytesToGB(args.worker_storage)
        if args.web_server_storage is not None:
            params['web_server_storage_gb'] = environments_api_util.MemorySizeBytesToGB(args.web_server_storage)
        if args.min_workers:
            params['min_workers'] = args.min_workers
        if args.max_workers:
            params['max_workers'] = args.max_workers
    self._addScheduledSnapshotFields(params, args, is_composer_v1)
    self._addTriggererFields(params, args, env_obj)
    if self._support_maintenance_window:
        params['clear_maintenance_window'] = args.clear_maintenance_window
        params['maintenance_window_start'] = args.maintenance_window_start
        params['maintenance_window_end'] = args.maintenance_window_end
        params['maintenance_window_recurrence'] = args.maintenance_window_recurrence
    params['airflow_database_retention_days'] = args.airflow_database_retention_days
    if args.enable_master_authorized_networks and args.disable_master_authorized_networks:
        raise command_util.InvalidUserInputError('Cannot specify --enable-master-authorized-networks with --disable-master-authorized-networks')
    if args.disable_master_authorized_networks and args.master_authorized_networks:
        raise command_util.InvalidUserInputError('Cannot specify --disable-master-authorized-networks with --master-authorized-networks')
    if args.enable_master_authorized_networks is None and args.master_authorized_networks:
        raise command_util.InvalidUserInputError('Cannot specify --master-authorized-networks without --enable-master-authorized-networks')
    if args.enable_master_authorized_networks or args.disable_master_authorized_networks:
        params['master_authorized_networks_enabled'] = True if args.enable_master_authorized_networks else False
    command_util.ValidateMasterAuthorizedNetworks(args.master_authorized_networks)
    params['master_authorized_networks'] = args.master_authorized_networks
    if args.enable_high_resilience or args.disable_high_resilience:
        if is_composer_v1:
            raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V1_ERROR_MSG.format(opt='enable_high_resilience' if args.enable_high_resilience else 'disable_high_resilience'))
        params['enable_high_resilience'] = bool(args.enable_high_resilience)
    if args.enable_logs_in_cloud_logging_only or args.disable_logs_in_cloud_logging_only:
        if is_composer_v1:
            raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V1_ERROR_MSG.format(opt='enable_logs_in_cloud_logging_only' if args.enable_logs_in_cloud_logging_only else 'disable_logs_in_cloud_logging_only'))
        params['enable_logs_in_cloud_logging_only'] = bool(args.enable_logs_in_cloud_logging_only)
    if args.enable_cloud_data_lineage_integration or args.disable_cloud_data_lineage_integration:
        if is_composer_v1:
            raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V1_ERROR_MSG.format(opt='enable-cloud-data-lineage-integration' if args.enable_cloud_data_lineage_integration else 'disable-cloud-data-lineage-integration'))
        params['cloud_data_lineage_integration_enabled'] = bool(args.enable_cloud_data_lineage_integration)
    if self._support_composer3flags:
        self._addComposer3Fields(params, args, env_obj)
    return patch_util.ConstructPatch(**params)