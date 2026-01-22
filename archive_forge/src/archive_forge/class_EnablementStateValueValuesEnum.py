from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnablementStateValueValuesEnum(_messages.Enum):
    """The enablement state of the custom module.

    Values:
      ENABLEMENT_STATE_UNSPECIFIED: Unspecified enablement state.
      ENABLED: The module is enabled at the given CRM resource.
      DISABLED: The module is disabled at the given CRM resource.
      INHERITED: State is inherited from an ancestor module. The module will
        either be effectively ENABLED or DISABLED based on its closest non-
        inherited ancestor module in the CRM hierarchy.
    """
    ENABLEMENT_STATE_UNSPECIFIED = 0
    ENABLED = 1
    DISABLED = 2
    INHERITED = 3