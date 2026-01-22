from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AcceleratorTypeValueValuesEnum(_messages.Enum):
    """Immutable. The type of accelerator(s) that may be attached to the
    machine as per accelerator_count.

    Values:
      ACCELERATOR_TYPE_UNSPECIFIED: Unspecified accelerator type, which means
        no accelerator.
      NVIDIA_TESLA_K80: Nvidia Tesla K80 GPU.
      NVIDIA_TESLA_P100: Nvidia Tesla P100 GPU.
      NVIDIA_TESLA_V100: Nvidia Tesla V100 GPU.
      NVIDIA_TESLA_P4: Nvidia Tesla P4 GPU.
      NVIDIA_TESLA_T4: Nvidia Tesla T4 GPU.
      NVIDIA_TESLA_A100: Nvidia Tesla A100 GPU.
      NVIDIA_A100_80GB: Nvidia A100 80GB GPU.
      NVIDIA_L4: Nvidia L4 GPU.
      NVIDIA_H100_80GB: Nvidia H100 80Gb GPU.
      TPU_V2: TPU v2.
      TPU_V3: TPU v3.
      TPU_V4_POD: TPU v4.
    """
    ACCELERATOR_TYPE_UNSPECIFIED = 0
    NVIDIA_TESLA_K80 = 1
    NVIDIA_TESLA_P100 = 2
    NVIDIA_TESLA_V100 = 3
    NVIDIA_TESLA_P4 = 4
    NVIDIA_TESLA_T4 = 5
    NVIDIA_TESLA_A100 = 6
    NVIDIA_A100_80GB = 7
    NVIDIA_L4 = 8
    NVIDIA_H100_80GB = 9
    TPU_V2 = 10
    TPU_V3 = 11
    TPU_V4_POD = 12