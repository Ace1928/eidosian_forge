from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryOrgunitsPatchRequest(_messages.Message):
    """A DirectoryOrgunitsPatchRequest object.

  Fields:
    customerId: Immutable ID of the G Suite account
    orgUnit: A OrgUnit resource to be passed as the request body.
    orgUnitPath: Full path of the organizational unit or its ID
  """
    customerId = _messages.StringField(1, required=True)
    orgUnit = _messages.MessageField('OrgUnit', 2)
    orgUnitPath = _messages.StringField(3, required=True)