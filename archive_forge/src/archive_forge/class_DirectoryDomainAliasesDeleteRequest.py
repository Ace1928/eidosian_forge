from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryDomainAliasesDeleteRequest(_messages.Message):
    """A DirectoryDomainAliasesDeleteRequest object.

  Fields:
    customer: Immutable ID of the G Suite account.
    domainAliasName: Name of domain alias to be retrieved.
  """
    customer = _messages.StringField(1, required=True)
    domainAliasName = _messages.StringField(2, required=True)