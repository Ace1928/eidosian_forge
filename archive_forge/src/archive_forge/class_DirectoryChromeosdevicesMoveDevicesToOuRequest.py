from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryChromeosdevicesMoveDevicesToOuRequest(_messages.Message):
    """A DirectoryChromeosdevicesMoveDevicesToOuRequest object.

  Fields:
    chromeOsMoveDevicesToOu: A ChromeOsMoveDevicesToOu resource to be passed
      as the request body.
    customerId: Immutable ID of the G Suite account
    orgUnitPath: Full path of the target organizational unit or its ID
  """
    chromeOsMoveDevicesToOu = _messages.MessageField('ChromeOsMoveDevicesToOu', 1)
    customerId = _messages.StringField(2, required=True)
    orgUnitPath = _messages.StringField(3, required=True)