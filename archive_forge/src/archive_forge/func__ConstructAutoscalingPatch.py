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
def _ConstructAutoscalingPatch(scheduler_cpu, worker_cpu, web_server_cpu, scheduler_memory_gb, worker_memory_gb, web_server_memory_gb, scheduler_storage_gb, worker_storage_gb, web_server_storage_gb, worker_min_count, worker_max_count, scheduler_count, release_track, triggerer_cpu, triggerer_memory_gb, triggerer_count, dag_processor_cpu, dag_processor_memory_gb, dag_processor_count, dag_processor_storage_gb):
    """Constructs an environment patch for Airflow web server machine type.

  Args:
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
    worker_min_count: int or None, minimum number of workers in the Environment.
      Can be specified only in Composer 2.0.0.
    worker_max_count: int or None, maximumn number of workers in the
      Environment. Can be specified only in Composer 2.0.0.
    scheduler_count: int or None, number of schedulers in the Environment. Can
      be specified only in Composer 2.0.0.
    release_track: base.ReleaseTrack, the release track of command. It dictates
      which Composer client library is used.
    triggerer_cpu: float or None, CPU allocated to Airflow triggerer. Can be
      specified only in Airflow 2.2.x and greater.
    triggerer_memory_gb: float or None, memory allocated to Airflow triggerer.
      Can be specified only in Airflow 2.2.x and greater.
    triggerer_count: int or None, number of triggerers in the Environment. Can
      be specified only in Airflow 2.2.x and greater
    dag_processor_cpu: float or None, CPU allocated to Airflow dag processor.
      Can be specified only in Composer 3.
    dag_processor_count: int or None, number of Airflow dag processors. Can be
      specified only in Composer 3.
    dag_processor_memory_gb: float or None, memory allocated to Airflow dag
      processor. Can be specified only in Composer 3.
    dag_processor_storage_gb: float or None, storage allocated to Airflow dag
      processor. Can be specified only in Composer 3.

  Returns:
    (str, Environment), the field mask and environment to use for update.
  """
    messages = api_util.GetMessagesModule(release_track=release_track)
    workload_resources = dict(scheduler=messages.SchedulerResource(cpu=scheduler_cpu, memoryGb=scheduler_memory_gb, storageGb=scheduler_storage_gb, count=scheduler_count), webServer=messages.WebServerResource(cpu=web_server_cpu, memoryGb=web_server_memory_gb, storageGb=web_server_storage_gb), worker=messages.WorkerResource(cpu=worker_cpu, memoryGb=worker_memory_gb, storageGb=worker_storage_gb, minCount=worker_min_count, maxCount=worker_max_count))
    if triggerer_count is not None or triggerer_cpu or triggerer_memory_gb:
        workload_resources['triggerer'] = messages.TriggererResource(cpu=triggerer_cpu, memoryGb=triggerer_memory_gb, count=triggerer_count)
    if release_track != base.ReleaseTrack.GA:
        if dag_processor_count is not None:
            workload_resources['dagProcessor'] = messages.DagProcessorResource(cpu=dag_processor_cpu, memoryGb=dag_processor_memory_gb, storageGb=dag_processor_storage_gb, count=dag_processor_count)
    config = messages.EnvironmentConfig(workloadsConfig=messages.WorkloadsConfig(**workload_resources))
    return ('config.workloads_config', messages.Environment(config=config))