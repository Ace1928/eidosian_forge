from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrderByValueValuesEnum(_messages.Enum):
    """Column to use for sorting results

    Values:
      email: Primary email of the user.
      familyName: User's family name.
      givenName: User's given name.
    """
    email = 0
    familyName = 1
    givenName = 2