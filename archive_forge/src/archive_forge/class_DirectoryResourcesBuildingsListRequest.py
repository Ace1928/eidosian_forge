from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryResourcesBuildingsListRequest(_messages.Message):
    """A DirectoryResourcesBuildingsListRequest object.

  Fields:
    customer: The unique ID for the customer's G Suite account. As an account
      administrator, you can also use the my_customer alias to represent your
      account's customer ID.
    maxResults: Maximum number of results to return.
    pageToken: Token to specify the next page in the list.
  """
    customer = _messages.StringField(1, required=True)
    maxResults = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)