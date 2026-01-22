from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryCustomersPatchRequest(_messages.Message):
    """A DirectoryCustomersPatchRequest object.

  Fields:
    customer: A Customer resource to be passed as the request body.
    customerKey: Id of the customer to be updated
  """
    customer = _messages.MessageField('Customer', 1)
    customerKey = _messages.StringField(2, required=True)