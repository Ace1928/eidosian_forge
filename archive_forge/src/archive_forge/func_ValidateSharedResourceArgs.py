from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ai import constants
def ValidateSharedResourceArgs(shared_resources_ref=None, machine_type=None, accelerator_dict=None, min_replica_count=None, max_replica_count=None, autoscaling_metric_specs=None):
    """Value validation for dedicated resource args while making a shared resource command call.

  Args:
      shared_resources_ref: str or None, the shared deployment resource pool
      full name the model should use, formatted as the full URI
      machine_type: str or None, the type of the machine to serve the model.
      accelerator_dict: dict or None, the accelerator attached to the deployed
        model from args.
      min_replica_count: int or None, the minimum number of replicas the
        deployed model will be always deployed on.
      max_replica_count: int or None, the maximum number of replicas the
        deployed model may be deployed on.
      autoscaling_metric_specs: dict or None, the metric specification that
        defines the target resource utilization for calculating the desired
        replica count.
  """
    if shared_resources_ref is None:
        return
    if machine_type is not None:
        raise exceptions.InvalidArgumentException('--machine-type', 'Cannot use\n    machine type and shared resources in the same command.')
    if accelerator_dict is not None:
        raise exceptions.InvalidArgumentException('--accelerator', 'Cannot\n    use accelerator and shared resources in the same command.')
    if min_replica_count is not None:
        raise exceptions.InvalidArgumentException('--max-replica-count', 'Cannot\n    use max replica count and shared resources in the same command.')
    if max_replica_count is not None:
        raise exceptions.InvalidArgumentException('--min-replica-count', 'Cannot\n    use min replica count and shared resources in the same command.')
    if autoscaling_metric_specs is not None:
        raise exceptions.InvalidArgumentException('--autoscaling-metric-specs', 'Cannot use autoscaling metric specs\n        and shared resources in the same command.')