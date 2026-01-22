from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1AcceleratorConfig(_messages.Message):
    """Represents a hardware accelerator request config. Note that the
  AcceleratorConfig can be used in both Jobs and Versions. Learn more about
  [accelerators for training](/ml-engine/docs/using-gpus) and [accelerators
  for online prediction](/ml-engine/docs/machine-types-online-
  prediction#gpus).

  Enums:
    TypeValueValuesEnum: The type of accelerator to use.

  Fields:
    count: The number of accelerators to attach to each machine running the
      job.
    type: The type of accelerator to use.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of accelerator to use.

    Values:
      ACCELERATOR_TYPE_UNSPECIFIED: Unspecified accelerator type. Default to
        no GPU.
      NVIDIA_TESLA_K80: Nvidia Tesla K80 GPU.
      NVIDIA_TESLA_P100: Nvidia Tesla P100 GPU.
      NVIDIA_TESLA_V100: Nvidia V100 GPU.
      NVIDIA_TESLA_P4: Nvidia Tesla P4 GPU.
      NVIDIA_TESLA_T4: Nvidia T4 GPU.
      NVIDIA_TESLA_A100: Nvidia A100 GPU.
      TPU_V2: TPU v2.
      TPU_V3: TPU v3.
      TPU_V2_POD: TPU v2 POD.
      TPU_V3_POD: TPU v3 POD.
      TPU_V4_POD: TPU v4 POD.
    """
        ACCELERATOR_TYPE_UNSPECIFIED = 0
        NVIDIA_TESLA_K80 = 1
        NVIDIA_TESLA_P100 = 2
        NVIDIA_TESLA_V100 = 3
        NVIDIA_TESLA_P4 = 4
        NVIDIA_TESLA_T4 = 5
        NVIDIA_TESLA_A100 = 6
        TPU_V2 = 7
        TPU_V3 = 8
        TPU_V2_POD = 9
        TPU_V3_POD = 10
        TPU_V4_POD = 11
    count = _messages.IntegerField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)