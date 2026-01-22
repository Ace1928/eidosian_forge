from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceRedistributionTypeValueValuesEnum(_messages.Enum):
    """The instance redistribution policy for regional managed instance
    groups. Valid values are: - PROACTIVE (default): The group attempts to
    maintain an even distribution of VM instances across zones in the region.
    - NONE: For non-autoscaled groups, proactive redistribution is disabled.

    Values:
      NONE: No action is being proactively performed in order to bring this
        IGM to its target instance distribution.
      PROACTIVE: This IGM will actively converge to its target instance
        distribution.
    """
    NONE = 0
    PROACTIVE = 1