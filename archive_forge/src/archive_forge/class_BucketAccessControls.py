from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BucketAccessControls(_messages.Message):
    """An access-control list.

  Fields:
    items: The list of items.
    kind: The kind of item this is. For lists of bucket access control
      entries, this is always storage#bucketAccessControls.
  """
    items = _messages.MessageField('BucketAccessControl', 1, repeated=True)
    kind = _messages.StringField(2, default=u'storage#bucketAccessControls')