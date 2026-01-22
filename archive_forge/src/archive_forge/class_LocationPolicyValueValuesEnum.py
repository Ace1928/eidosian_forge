from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocationPolicyValueValuesEnum(_messages.Enum):
    """Location policy used when scaling up a nodepool.

    Values:
      LOCATION_POLICY_UNSPECIFIED: Not set.
      BALANCED: BALANCED is a best effort policy that aims to balance the
        sizes of different zones.
      ANY: ANY policy picks zones that have the highest capacity available.
    """
    LOCATION_POLICY_UNSPECIFIED = 0
    BALANCED = 1
    ANY = 2