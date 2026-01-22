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
def _addTriggererFields(self, params, args, env_obj):
    triggerer_supported = image_versions_command_util.IsVersionTriggererCompatible(env_obj.config.softwareConfig.imageVersion)
    triggerer_count = None
    triggerer_cpu = None
    triggerer_memory_gb = None
    if env_obj.config.workloadsConfig and env_obj.config.workloadsConfig.triggerer and (env_obj.config.workloadsConfig.triggerer.count != 0):
        triggerer_count = env_obj.config.workloadsConfig.triggerer.count
        triggerer_memory_gb = env_obj.config.workloadsConfig.triggerer.memoryGb
        triggerer_cpu = env_obj.config.workloadsConfig.triggerer.cpu
    if args.disable_triggerer or args.enable_triggerer:
        triggerer_count = 1 if args.enable_triggerer else 0
    if args.triggerer_count is not None:
        triggerer_count = args.triggerer_count
    if args.triggerer_cpu:
        triggerer_cpu = args.triggerer_cpu
    if args.triggerer_memory:
        triggerer_memory_gb = environments_api_util.MemorySizeBytesToGB(args.triggerer_memory)
    possible_args = {'triggerer-count': args.enable_triggerer, 'triggerer-cpu': args.triggerer_cpu, 'triggerer-memory': args.triggerer_memory}
    for k, v in possible_args.items():
        if v and (not triggerer_supported):
            raise command_util.InvalidUserInputError(flags.INVALID_OPTION_FOR_MIN_IMAGE_VERSION_ERROR_MSG.format(opt=k, composer_version=flags.MIN_TRIGGERER_COMPOSER_VERSION, airflow_version=flags.MIN_TRIGGERER_AIRFLOW_VERSION))
    if not triggerer_count:
        if args.triggerer_cpu:
            raise command_util.InvalidUserInputError('Cannot specify --triggerer-cpu without enabled triggerer')
        if args.triggerer_memory:
            raise command_util.InvalidUserInputError('Cannot specify --triggerer-memory without enabled triggerer')
    if triggerer_count == 1 and (not (triggerer_memory_gb and triggerer_cpu)):
        raise command_util.InvalidUserInputError('Cannot enable triggerer without providing triggerer memory and cpu.')
    params['triggerer_count'] = triggerer_count
    if triggerer_count:
        params['triggerer_cpu'] = triggerer_cpu
        params['triggerer_memory_gb'] = triggerer_memory_gb