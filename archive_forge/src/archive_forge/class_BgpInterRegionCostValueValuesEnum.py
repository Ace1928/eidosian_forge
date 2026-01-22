from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BgpInterRegionCostValueValuesEnum(_messages.Enum):
    """Allows to define a preferred approach for handling inter-region cost
    in the selection process when using the STANDARD BGP best path selection
    algorithm. Can be DEFAULT or ADD_COST_TO_MED.

    Values:
      ADD_COST_TO_MED: <no description>
      DEFAULT: <no description>
    """
    ADD_COST_TO_MED = 0
    DEFAULT = 1