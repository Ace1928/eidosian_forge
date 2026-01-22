from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalNrqzValidation(_messages.Message):
    """Information about National Radio Quiet Zone validation.

  Enums:
    StateValueValuesEnum: State of the NRQZ validation info.

  Fields:
    caseId: Validation case ID.
    cpiId: CPI who signed the validation.
    latitude: Device latitude that's associated with the validation.
    longitude: Device longitude that's associated with the validation.
    state: State of the NRQZ validation info.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of the NRQZ validation info.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      DRAFT: Draft state.
      FINAL: Final state.
    """
        STATE_UNSPECIFIED = 0
        DRAFT = 1
        FINAL = 2
    caseId = _messages.StringField(1)
    cpiId = _messages.StringField(2)
    latitude = _messages.FloatField(3)
    longitude = _messages.FloatField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)