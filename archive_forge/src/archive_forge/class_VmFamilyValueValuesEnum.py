from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmFamilyValueValuesEnum(_messages.Enum):
    """The VM family that all instances scheduled against this reservation
    must belong to.

    Values:
      VM_FAMILY_CLOUD_TPU_LITE_DEVICE_CT5L: <no description>
      VM_FAMILY_CLOUD_TPU_LITE_POD_SLICE_CT5LP: <no description>
      VM_FAMILY_CLOUD_TPU_POD_SLICE_CT4P: <no description>
    """
    VM_FAMILY_CLOUD_TPU_LITE_DEVICE_CT5L = 0
    VM_FAMILY_CLOUD_TPU_LITE_POD_SLICE_CT5LP = 1
    VM_FAMILY_CLOUD_TPU_POD_SLICE_CT4P = 2