from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserPhoto(_messages.Message):
    """JSON template for Photo object in Directory API.

  Fields:
    etag: ETag of the resource.
    height: Height in pixels of the photo
    id: Unique identifier of User (Read-only)
    kind: Kind of resource this is.
    mimeType: Mime Type of the photo
    photoData: Base64 encoded photo data
    primaryEmail: Primary email of User (Read-only)
    width: Width in pixels of the photo
  """
    etag = _messages.StringField(1)
    height = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    id = _messages.StringField(3)
    kind = _messages.StringField(4, default=u'admin#directory#user#photo')
    mimeType = _messages.StringField(5)
    photoData = _messages.BytesField(6)
    primaryEmail = _messages.StringField(7)
    width = _messages.IntegerField(8, variant=_messages.Variant.INT32)