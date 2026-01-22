from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MobileDevices(_messages.Message):
    """JSON response template for List Mobile Devices operation in Directory

  API.

  Fields:
    etag: ETag of the resource.
    kind: Kind of resource this is.
    mobiledevices: List of Mobile Device objects.
    nextPageToken: Token used to access next page of this result.
  """
    etag = _messages.StringField(1)
    kind = _messages.StringField(2, default=u'admin#directory#mobiledevices')
    mobiledevices = _messages.MessageField('MobileDevice', 3, repeated=True)
    nextPageToken = _messages.StringField(4)