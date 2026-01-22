from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadProfileValueValuesEnum(_messages.Enum):
    """The workload profile for the volume.

    Values:
      WORKLOAD_PROFILE_UNSPECIFIED: The workload profile is in an unknown
        state.
      GENERIC: The workload profile is generic.
      HANA: The workload profile is hana.
    """
    WORKLOAD_PROFILE_UNSPECIFIED = 0
    GENERIC = 1
    HANA = 2