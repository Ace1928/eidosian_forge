from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import validation
def _ValidateResourcePoolSpecsFromConfig(resource_pools, version):
    """Validate ResourcePoolSpec message instances imported from the config file."""
    if not resource_pools:
        raise exceptions.InvalidArgumentException('--config', 'At least one [resourcePools] required in but not provided in config.')
    for spec in resource_pools:
        if not spec.machineSpec:
            raise exceptions.InvalidArgumentException('--config', 'Field [machineSpec] required in but not provided in config.')
        if not spec.machineSpec.machineType:
            raise exceptions.InvalidArgumentException('--config', 'Field [machineType] required in but not provided in config.')
        if not spec.replicaCount and (not spec.autoscalingSpec):
            raise exceptions.InvalidArgumentException('--config', 'Field [replicaCount] required in but not provided in config.')
        if spec.autoscalingSpec and (not spec.autoscalingSpec.minReplicaCount):
            raise exceptions.InvalidArgumentException('--config', 'Field [minReplicaCount] required when using autoscaling')
        if spec.autoscalingSpec and (not spec.autoscalingSpec.maxReplicaCount):
            raise exceptions.InvalidArgumentException('--config', 'Field [maxReplicaCount] required when using autoscaling')
        if spec.machineSpec.acceleratorCount and (not spec.machineSpec.acceleratorType):
            raise exceptions.InvalidArgumentException('--config', 'Field [acceleratorType] required as [acceleratorCount] is specifiedin config.')
        if spec.diskSpec and (spec.diskSpec.bootDiskSizeGb and (not spec.diskSpec.bootDiskType)):
            raise exceptions.InvalidArgumentException('--config', 'Field [bootDiskType] required as [bootDiskSizeGb] is specifiedin config.')
        if spec.machineSpec.acceleratorType:
            accelerator_type = str(spec.machineSpec.acceleratorType.name)
            type_enum = api_util.GetMessage('MachineSpec', version).AcceleratorTypeValueValuesEnum
            valid_types = [type for type in type_enum.names() if type.startswith('NVIDIA')]
            if accelerator_type not in valid_types:
                raise exceptions.InvalidArgumentException('--config', 'Found invalid value of [acceleratorType]: {actual}. Available values are [{expected}].'.format(actual=accelerator_type, expected=', '.join((v for v in sorted(valid_types)))))