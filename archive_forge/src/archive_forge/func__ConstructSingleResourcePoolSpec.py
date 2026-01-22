from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def _ConstructSingleResourcePoolSpec(aiplatform_client, spec):
    """Constructs a single resource pool spec.

  Args:
    aiplatform_client: The AI Platform API client used.
    spec: A dict whose fields represent a resource pool spec.

  Returns:
    A ResourcePoolSpec message instance for setting a resource pool in a
    Persistent Resource
  """
    resource_pool = aiplatform_client.GetMessage('ResourcePool')()
    machine_spec_msg = aiplatform_client.GetMessage('MachineSpec')
    machine_spec = machine_spec_msg(machineType=spec.get('machine-type'))
    accelerator_type = spec.get('accelerator-type')
    if accelerator_type:
        machine_spec.acceleratorType = arg_utils.ChoiceToEnum(accelerator_type, machine_spec_msg.AcceleratorTypeValueValuesEnum)
        machine_spec.acceleratorCount = int(spec.get('accelerator-count', 1))
    resource_pool.machineSpec = machine_spec
    replica_count = spec.get('replica-count')
    if replica_count:
        resource_pool.replicaCount = int(replica_count)
    min_replica_count = spec.get('min-replica-count')
    max_replica_count = spec.get('max-replica-count')
    if min_replica_count or max_replica_count:
        autoscaling_spec = aiplatform_client.GetMessage('ResourcePoolAutoscalingSpec')()
        autoscaling_spec.minReplicaCount = int(min_replica_count)
        autoscaling_spec.maxReplicaCount = int(max_replica_count)
        resource_pool.autoscalingSpec = autoscaling_spec
    disk_type = spec.get('disk-type')
    disk_size = spec.get('disk-size')
    if disk_type:
        disk_spec_msg = aiplatform_client.GetMessage('DiskSpec')
        disk_spec = disk_spec_msg(bootDiskType=disk_type, bootDiskSizeGb=disk_size)
        resource_pool.diskSpec = disk_spec
    return resource_pool