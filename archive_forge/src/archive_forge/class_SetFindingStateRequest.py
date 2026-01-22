from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetFindingStateRequest(_messages.Message):
    """Request message for updating a finding's state.

  Enums:
    StateValueValuesEnum: Required. The desired State of the finding.

  Fields:
    state: Required. The desired State of the finding.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Required. The desired State of the finding.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      ACTIVE: The finding requires attention and has not been addressed yet.
      INACTIVE: The finding has been fixed, triaged as a non-issue or
        otherwise addressed and is no longer active.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        INACTIVE = 2
    state = _messages.EnumField('StateValueValuesEnum', 1)