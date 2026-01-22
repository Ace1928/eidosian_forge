from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainAliases(_messages.Message):
    """JSON response template to list domain aliases in Directory API.

  Fields:
    domainAliases: List of domain alias objects.
    etag: ETag of the resource.
    kind: Kind of resource this is.
  """
    domainAliases = _messages.MessageField('DomainAlias', 1, repeated=True)
    etag = _messages.StringField(2)
    kind = _messages.StringField(3, default=u'admin#directory#domainAliases')