from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainAlias(_messages.Message):
    """JSON template for Domain Alias object in Directory API.

  Fields:
    creationTime: The creation time of the domain alias. (Read-only).
    domainAliasName: The domain alias name.
    etag: ETag of the resource.
    kind: Kind of resource this is.
    parentDomainName: The parent domain name that the domain alias is
      associated with. This can either be a primary or secondary domain name
      within a customer.
    verified: Indicates the verification state of a domain alias. (Read-only)
  """
    creationTime = _messages.IntegerField(1)
    domainAliasName = _messages.StringField(2)
    etag = _messages.StringField(3)
    kind = _messages.StringField(4, default=u'admin#directory#domainAlias')
    parentDomainName = _messages.StringField(5)
    verified = _messages.BooleanField(6)