from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CardLabelsValueListEntryValuesEnum(_messages.Enum):
    """CardLabelsValueListEntryValuesEnum enum type.

    Values:
      CARD_LABEL_UNSPECIFIED: No label specified.
      PREPAID: This card has been detected as prepaid.
      VIRTUAL: This card has been detected as virtual, such as a card number
        generated for a single transaction or merchant.
      UNEXPECTED_LOCATION: This card has been detected as being used in an
        unexpected geographic location.
    """
    CARD_LABEL_UNSPECIFIED = 0
    PREPAID = 1
    VIRTUAL = 2
    UNEXPECTED_LOCATION = 3