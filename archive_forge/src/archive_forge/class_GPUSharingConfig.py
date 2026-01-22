from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GPUSharingConfig(_messages.Message):
    """GPUSharingConfig represents the GPU sharing configuration for Hardware
  Accelerators.

  Enums:
    GpuSharingStrategyValueValuesEnum: The type of GPU sharing strategy to
      enable on the GPU node.

  Fields:
    gpuSharingStrategy: The type of GPU sharing strategy to enable on the GPU
      node.
    maxSharedClientsPerGpu: The max number of containers that can share a
      physical GPU.
  """

    class GpuSharingStrategyValueValuesEnum(_messages.Enum):
        """The type of GPU sharing strategy to enable on the GPU node.

    Values:
      GPU_SHARING_STRATEGY_UNSPECIFIED: Default value.
      TIME_SHARING: GPUs are time-shared between containers.
      MPS: GPUs are shared between containers with NVIDIA MPS.
    """
        GPU_SHARING_STRATEGY_UNSPECIFIED = 0
        TIME_SHARING = 1
        MPS = 2
    gpuSharingStrategy = _messages.EnumField('GpuSharingStrategyValueValuesEnum', 1)
    maxSharedClientsPerGpu = _messages.IntegerField(2)