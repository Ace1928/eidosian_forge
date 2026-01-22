from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Privileges(_messages.Message):
    """JSON response template for List privileges operation in Directory API.

  Fields:
    etag: ETag of the resource.
    items: A list of Privilege resources.
    kind: The type of the API resource. This is always
      admin#directory#privileges.
  """
    etag = _messages.StringField(1)
    items = _messages.MessageField('Privilege', 2, repeated=True)
    kind = _messages.StringField(3, default=u'admin#directory#privileges')