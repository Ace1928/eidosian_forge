from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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