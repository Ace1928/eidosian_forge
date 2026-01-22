from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryCustomersGetRequest(_messages.Message):
    """A DirectoryCustomersGetRequest object.

  Fields:
    customerKey: Id of the customer to be retrieved
  """
    customerKey = _messages.StringField(1, required=True)