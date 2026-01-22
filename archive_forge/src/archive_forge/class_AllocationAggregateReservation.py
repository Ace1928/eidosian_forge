from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllocationAggregateReservation(_messages.Message):
    """This reservation type is specified by total resource amounts (e.g. total
  count of CPUs) and can account for multiple instance SKUs. In other words,
  one can create instances of varying shapes against this reservation.

  Enums:
    VmFamilyValueValuesEnum: The VM family that all instances scheduled
      against this reservation must belong to.
    WorkloadTypeValueValuesEnum: The workload type of the instances that will
      target this reservation.

  Fields:
    inUseResources: [Output only] List of resources currently in use.
    reservedResources: List of reserved resources (CPUs, memory,
      accelerators).
    vmFamily: The VM family that all instances scheduled against this
      reservation must belong to.
    workloadType: The workload type of the instances that will target this
      reservation.
  """

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

    class WorkloadTypeValueValuesEnum(_messages.Enum):
        """The workload type of the instances that will target this reservation.

    Values:
      BATCH: Reserved resources will be optimized for BATCH workloads, such as
        ML training.
      SERVING: Reserved resources will be optimized for SERVING workloads,
        such as ML inference.
      UNSPECIFIED: <no description>
    """
        BATCH = 0
        SERVING = 1
        UNSPECIFIED = 2
    inUseResources = _messages.MessageField('AllocationAggregateReservationReservedResourceInfo', 1, repeated=True)
    reservedResources = _messages.MessageField('AllocationAggregateReservationReservedResourceInfo', 2, repeated=True)
    vmFamily = _messages.EnumField('VmFamilyValueValuesEnum', 3)
    workloadType = _messages.EnumField('WorkloadTypeValueValuesEnum', 4)