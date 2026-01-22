from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
def ConstructPatch(is_composer_v1, env_ref=None, node_count=None, update_pypi_packages_from_file=None, clear_pypi_packages=None, remove_pypi_packages=None, update_pypi_packages=None, clear_labels=None, remove_labels=None, update_labels=None, clear_airflow_configs=None, remove_airflow_configs=None, update_airflow_configs=None, clear_env_variables=None, remove_env_variables=None, update_env_variables=None, update_image_version=None, update_web_server_access_control=None, cloud_sql_machine_type=None, web_server_machine_type=None, scheduler_cpu=None, worker_cpu=None, web_server_cpu=None, scheduler_memory_gb=None, worker_memory_gb=None, web_server_memory_gb=None, scheduler_storage_gb=None, worker_storage_gb=None, web_server_storage_gb=None, min_workers=None, max_workers=None, scheduler_count=None, clear_maintenance_window=None, maintenance_window_start=None, maintenance_window_end=None, maintenance_window_recurrence=None, environment_size=None, master_authorized_networks_enabled=None, master_authorized_networks=None, airflow_database_retention_days=None, release_track=base.ReleaseTrack.GA, triggerer_cpu=None, triggerer_memory_gb=None, triggerer_count=None, enable_scheduled_snapshot_creation=None, snapshot_location=None, snapshot_schedule_timezone=None, snapshot_creation_schedule=None, cloud_data_lineage_integration_enabled=None, support_web_server_plugins=None, support_private_builds_only=None, dag_processor_cpu=None, dag_processor_count=None, dag_processor_memory_gb=None, dag_processor_storage_gb=None, disable_vpc_connectivity=None, network=None, subnetwork=None, network_attachment=None, workload_updated=None, enable_private_environment=None, disable_private_environment=None, enable_high_resilience=None, enable_logs_in_cloud_logging_only=None):
    """Constructs an environment patch.

  Args:
    is_composer_v1: boolean representing if patch request is for Composer 1.*.*
      Environment.
    env_ref: resource argument, Environment resource argument for environment
      being updated.
    node_count: int, the desired node count
    update_pypi_packages_from_file: str, path to local requirements file
      containing desired pypi dependencies.
    clear_pypi_packages: bool, whether to uninstall all PyPI packages.
    remove_pypi_packages: iterable(string), Iterable of PyPI packages to
      uninstall.
    update_pypi_packages: {string: string}, dict mapping PyPI package name to
      extras and version specifier.
    clear_labels: bool, whether to clear the labels dictionary.
    remove_labels: iterable(string), Iterable of label names to remove.
    update_labels: {string: string}, dict of label names and values to set.
    clear_airflow_configs: bool, whether to clear the Airflow configs
      dictionary.
    remove_airflow_configs: iterable(string), Iterable of Airflow config
      property names to remove.
    update_airflow_configs: {string: string}, dict of Airflow config property
      names and values to set.
    clear_env_variables: bool, whether to clear the environment variables
      dictionary.
    remove_env_variables: iterable(string), Iterable of environment variables to
      remove.
    update_env_variables: {string: string}, dict of environment variable names
      and values to set.
    update_image_version: string, image version to use for environment upgrade
    update_web_server_access_control: [{string: string}], Webserver access
      control to set
    cloud_sql_machine_type: str or None, Cloud SQL machine type used by the
      Airflow database.
    web_server_machine_type: str or None, machine type used by the Airflow web
      server
    scheduler_cpu: float or None, CPU allocated to Airflow scheduler. Can be
      specified only in Composer 2.0.0.
    worker_cpu: float or None, CPU allocated to each Airflow worker. Can be
      specified only in Composer 2.0.0.
    web_server_cpu: float or None, CPU allocated to Airflow web server. Can be
      specified only in Composer 2.0.0.
    scheduler_memory_gb: float or None, memory allocated to Airflow scheduler.
      Can be specified only in Composer 2.0.0.
    worker_memory_gb: float or None, memory allocated to each Airflow worker.
      Can be specified only in Composer 2.0.0.
    web_server_memory_gb: float or None, memory allocated to Airflow web server.
      Can be specified only in Composer 2.0.0.
    scheduler_storage_gb: float or None, storage allocated to Airflow scheduler.
      Can be specified only in Composer 2.0.0.
    worker_storage_gb: float or None, storage allocated to each Airflow worker.
      Can be specified only in Composer 2.0.0.
    web_server_storage_gb: float or None, storage allocated to Airflow web
      server. Can be specified only in Composer 2.0.0.
    min_workers: int or None, minimum number of workers in the Environment. Can
      be specified only in Composer 2.0.0.
    max_workers: int or None, maximumn number of workers in the Environment. Can
      be specified only in Composer 2.0.0.
    scheduler_count: int or None, number of schedulers in the Environment. Can
      be specified only in Composer 2.0.0.
    clear_maintenance_window: bool or None, specifies if maintenance window
      options should be cleared.
    maintenance_window_start: Datetime or None, a starting date of the
      maintenance window.
    maintenance_window_end: Datetime or None, an ending date of the maintenance
      window.
    maintenance_window_recurrence: str or None, recurrence RRULE for the
      maintenance window.
    environment_size: str or None, one of small, medium and large.
    master_authorized_networks_enabled: bool or None, whether the feature should
      be enabled
    master_authorized_networks: iterable(string) or None, iterable of master
      authorized networks.
    airflow_database_retention_days: Optional[int], the number of retention days
      for airflow database data retention mechanism. Infinite retention will be
      applied in case `0` or no integer is provided.
    release_track: base.ReleaseTrack, the release track of command. Will dictate
      which Composer client library will be used.
    triggerer_cpu: float or None, CPU allocated to Airflow triggerer. Can be
      specified only in Airflow 2.2.x and greater.
    triggerer_memory_gb: float or None, memory allocated to Airflow triggerer.
      Can be specified only in Airflow 2.2.x and greater.
    triggerer_count: int or None, number of triggerers in the Environment. Can
      be specified only in Airflow 2.2.x and greater
    enable_scheduled_snapshot_creation: bool, whether the automatic snapshot
      creation should be enabled
    snapshot_location: str, a Cloud Storage location used to store automatically
      created snapshots
    snapshot_schedule_timezone: str, time zone that sets the context to
      interpret snapshot_creation_schedule.
    snapshot_creation_schedule: str, cron expression that specifies when
      snapshots will be created
    cloud_data_lineage_integration_enabled: bool or None, whether the feature
      should be enabled
    support_web_server_plugins: bool or None, whether to enable/disable the
      support for web server plugins
    support_private_builds_only: bool or None, whether to enable/disable the
      support for private only builds
    dag_processor_cpu: float or None, CPU allocated to Airflow dag processor.
      Can be specified only in Composer 3.
    dag_processor_count: int or None, number of Airflow dag processors. Can be
      specified only in Composer 3.
    dag_processor_memory_gb: float or None, memory allocated to Airflow dag
      processor. Can be specified only in Composer 3.
    dag_processor_storage_gb: float or None, storage allocated to Airflow dag
      processor. Can be specified only in Composer 3.
    disable_vpc_connectivity: bool or None, defines whether to disable
      connectivity with a user's VPC network. Can be specified only in Composer
      3.
    network: str or None, the Compute Engine network to which to connect the
      environment specified as relative resource name. Can be specified only in
      Composer 3.
    subnetwork: str or None, the Compute Engine subnetwork to which to connect
      the environment specified as relative resource name. Can be specified only
      in Composer 3.
    network_attachment: str or None, the Compute Engine network attachment that
      is used as PSC Network entry point.
    workload_updated: bool or None, verify if workload config has been updated
    enable_private_environment: bool or None, defines whether the internet
      access is disabled from Composer components. Can be specified only in
      Composer 3.
    disable_private_environment: bool or None, defines whether the internet
      access is enabled from Composer components. Can be specified only in
      Composer 3.
    enable_high_resilience: bool or None, defines whether high resilience should
      be enabled for given environment. Can be specified only in Composer 2.
    enable_logs_in_cloud_logging_only: bool or None, defines whether logs in
      cloud logging only feature should be enabled for given environment. Can be
      specified only in composer 2.

  Returns:
    (str, Environment), the field mask and environment to use for update.

  Raises:
    command_util.Error: if no update type is specified
  """
    if node_count:
        return _ConstructNodeCountPatch(node_count, release_track=release_track)
    if environment_size:
        return _ConstructEnvironmentSizePatch(environment_size, release_track=release_track)
    if update_pypi_packages_from_file:
        return _ConstructPyPiPackagesPatch(True, [], command_util.ParseRequirementsFile(update_pypi_packages_from_file), release_track=release_track)
    if clear_pypi_packages or remove_pypi_packages or update_pypi_packages:
        return _ConstructPyPiPackagesPatch(clear_pypi_packages, remove_pypi_packages, update_pypi_packages, release_track=release_track)
    if enable_private_environment or disable_private_environment:
        return _ConstructPrivateEnvironmentPatch(enable_private_environment, release_track=release_track)
    if clear_labels or remove_labels or update_labels:
        return _ConstructLabelsPatch(clear_labels, remove_labels, update_labels, release_track=release_track)
    if clear_airflow_configs or remove_airflow_configs or update_airflow_configs:
        return _ConstructAirflowConfigsPatch(clear_airflow_configs, remove_airflow_configs, update_airflow_configs, release_track=release_track)
    if clear_env_variables or remove_env_variables or update_env_variables:
        return _ConstructEnvVariablesPatch(env_ref, clear_env_variables, remove_env_variables, update_env_variables, release_track=release_track)
    if update_image_version:
        return _ConstructImageVersionPatch(update_image_version, release_track=release_track)
    if update_web_server_access_control is not None:
        return _ConstructWebServerAccessControlPatch(update_web_server_access_control, release_track=release_track)
    if cloud_sql_machine_type:
        return _ConstructCloudSqlMachineTypePatch(cloud_sql_machine_type, release_track=release_track)
    if web_server_machine_type:
        return _ConstructWebServerMachineTypePatch(web_server_machine_type, release_track=release_track)
    if master_authorized_networks_enabled is not None:
        return _ConstructMasterAuthorizedNetworksTypePatch(master_authorized_networks_enabled, master_authorized_networks, release_track)
    if enable_scheduled_snapshot_creation is not None:
        return _ConstructScheduledSnapshotPatch(enable_scheduled_snapshot_creation, snapshot_creation_schedule, snapshot_location, snapshot_schedule_timezone, release_track)
    if support_private_builds_only is not None:
        return _ConstructPrivateBuildsOnlyPatch(support_private_builds_only, release_track)
    if support_web_server_plugins is not None:
        return _ConstructWebServerPluginsModePatch(support_web_server_plugins, release_track)
    if disable_vpc_connectivity is not None or network or subnetwork or network_attachment:
        return _ConstructVpcConnectivityPatch(disable_vpc_connectivity, network, subnetwork, network_attachment, release_track)
    if airflow_database_retention_days is not None:
        return _ConstructAirflowDatabaseRetentionDaysPatch(airflow_database_retention_days, release_track)
    if is_composer_v1 and scheduler_count:
        return _ConstructSoftwareConfigurationSchedulerCountPatch(scheduler_count=scheduler_count, release_track=release_track)
    if workload_updated:
        if is_composer_v1:
            raise command_util.Error('You cannot use Workloads Config flags introduced in Composer 2.X when updating Composer 1.X environments.')
        else:
            return _ConstructAutoscalingPatch(scheduler_cpu=scheduler_cpu, worker_cpu=worker_cpu, web_server_cpu=web_server_cpu, scheduler_memory_gb=scheduler_memory_gb, worker_memory_gb=worker_memory_gb, web_server_memory_gb=web_server_memory_gb, scheduler_storage_gb=scheduler_storage_gb, worker_storage_gb=worker_storage_gb, web_server_storage_gb=web_server_storage_gb, worker_min_count=min_workers, worker_max_count=max_workers, scheduler_count=scheduler_count, release_track=release_track, triggerer_cpu=triggerer_cpu, triggerer_memory_gb=triggerer_memory_gb, triggerer_count=triggerer_count, dag_processor_cpu=dag_processor_cpu, dag_processor_memory_gb=dag_processor_memory_gb, dag_processor_count=dag_processor_count, dag_processor_storage_gb=dag_processor_storage_gb)
    if maintenance_window_start and maintenance_window_end and maintenance_window_recurrence or clear_maintenance_window:
        return _ConstructMaintenanceWindowPatch(maintenance_window_start, maintenance_window_end, maintenance_window_recurrence, clear_maintenance_window, release_track=release_track)
    if cloud_data_lineage_integration_enabled is not None:
        return _ConstructSoftwareConfigurationCloudDataLineageIntegrationPatch(cloud_data_lineage_integration_enabled, release_track)
    if enable_high_resilience is not None:
        return _ConstructHighResiliencePatch(enable_high_resilience, release_track)
    if enable_logs_in_cloud_logging_only is not None:
        return _ConstructLogsInCloudLoggingOnlyPatch(enable_logs_in_cloud_logging_only, release_track)
    raise command_util.Error('Cannot update Environment with no update type specified.')