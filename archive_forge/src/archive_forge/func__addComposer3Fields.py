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
def _addComposer3Fields(self, params, args, env_obj):
    is_composer3 = image_versions_command_util.IsVersionComposer3Compatible(env_obj.config.softwareConfig.imageVersion)
    possible_args = {'support-web-server-plugins': args.support_web_server_plugins, 'enable-private-builds-only': args.enable_private_builds_only, 'disable-private-builds-only': args.disable_private_builds_only, 'dag-processor-cpu': args.dag_processor_cpu, 'dag-processor-memory': args.dag_processor_memory, 'dag-processor-count': args.dag_processor_count, 'dag-processor-storage': args.dag_processor_storage, 'disable-vpc-connectivity': args.disable_vpc_connectivity, 'enable-private-environment': args.enable_private_environment, 'disable-private-environment': args.disable_private_environment, 'network': args.network, 'subnetwork': args.subnetwork, 'clear-maintenance-window': args.clear_maintenance_window}
    for k, v in possible_args.items():
        if v is not None and (not is_composer3):
            raise command_util.InvalidUserInputError(flags.COMPOSER3_IS_REQUIRED_MSG.format(opt=k, composer_version=flags.MIN_COMPOSER3_VERSION))
    if args.dag_processor_count is not None or args.dag_processor_cpu or args.dag_processor_memory or args.dag_processor_storage:
        params['workload_updated'] = True
    dag_processor_count = None
    dag_processor_cpu = None
    dag_processor_memory_gb = None
    dag_processor_storage_gb = None
    if env_obj.config.workloadsConfig and env_obj.config.workloadsConfig.dagProcessor:
        dag_processor_resource = env_obj.config.workloadsConfig.dagProcessor
        dag_processor_count = dag_processor_resource.count
        dag_processor_cpu = dag_processor_resource.cpu
        dag_processor_memory_gb = dag_processor_resource.memoryGb
        dag_processor_storage_gb = dag_processor_resource.storageGb
    if args.dag_processor_count is not None:
        dag_processor_count = args.dag_processor_count
    if args.dag_processor_cpu:
        dag_processor_cpu = args.dag_processor_cpu
    if args.dag_processor_memory:
        dag_processor_memory_gb = environments_api_util.MemorySizeBytesToGB(args.dag_processor_memory)
    if args.dag_processor_storage:
        dag_processor_storage_gb = environments_api_util.MemorySizeBytesToGB(args.dag_processor_storage)
    if args.support_web_server_plugins is not None:
        params['support_web_server_plugins'] = args.support_web_server_plugins
    if args.enable_private_builds_only or args.disable_private_builds_only:
        params['support_private_builds_only'] = True if args.enable_private_builds_only else False
    if args.enable_private_environment is not None:
        params['enable_private_environment'] = args.enable_private_environment
    if args.disable_private_environment is not None:
        params['disable_private_environment'] = args.disable_private_environment
    params['dag_processor_count'] = dag_processor_count
    params['dag_processor_cpu'] = dag_processor_cpu
    params['dag_processor_memory_gb'] = dag_processor_memory_gb
    params['dag_processor_storage_gb'] = dag_processor_storage_gb
    if args.disable_vpc_connectivity:
        params['disable_vpc_connectivity'] = True
    if args.network_attachment:
        params['network_attachment'] = args.network_attachment
    if args.network:
        params['network'] = args.network
    if args.subnetwork:
        params['subnetwork'] = args.subnetwork