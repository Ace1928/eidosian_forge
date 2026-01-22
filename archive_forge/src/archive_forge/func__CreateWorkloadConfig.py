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
def _CreateWorkloadConfig(messages, flags):
    """Creates workload config from parameters."""
    workload_resources = dict(scheduler=messages.SchedulerResource(cpu=flags.scheduler_cpu, memoryGb=flags.scheduler_memory_gb, storageGb=flags.scheduler_storage_gb, count=flags.scheduler_count), webServer=messages.WebServerResource(cpu=flags.web_server_cpu, memoryGb=flags.web_server_memory_gb, storageGb=flags.web_server_storage_gb), worker=messages.WorkerResource(cpu=flags.worker_cpu, memoryGb=flags.worker_memory_gb, storageGb=flags.worker_storage_gb, minCount=flags.min_workers, maxCount=flags.max_workers))
    if flags.enable_triggerer or flags.triggerer_cpu or flags.triggerer_memory_gb or (flags.triggerer_count is not None):
        triggerer_count = 1 if flags.enable_triggerer else 0
        if flags.triggerer_count is not None:
            triggerer_count = flags.triggerer_count
        workload_resources['triggerer'] = messages.TriggererResource(cpu=flags.triggerer_cpu, memoryGb=flags.triggerer_memory_gb, count=triggerer_count)
    if flags.dag_processor_cpu or flags.dag_processor_count is not None or flags.dag_processor_memory_gb or flags.dag_processor_storage_gb:
        workload_resources['dagProcessor'] = messages.DagProcessorResource(cpu=flags.dag_processor_cpu, memoryGb=flags.dag_processor_memory_gb, storageGb=flags.dag_processor_storage_gb, count=flags.dag_processor_count)
    return messages.WorkloadsConfig(**workload_resources)