from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrgUnits(_messages.Message):
    """JSON response template for List Organization Units operation in

  Directory API.

  Fields:
    etag: ETag of the resource.
    kind: Kind of resource this is.
    organizationUnits: List of user objects.
  """
    etag = _messages.StringField(1)
    kind = _messages.StringField(2, default=u'admin#directory#orgUnits')
    organizationUnits = _messages.MessageField('OrgUnit', 3, repeated=True)