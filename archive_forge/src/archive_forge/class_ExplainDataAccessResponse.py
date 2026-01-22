from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExplainDataAccessResponse(_messages.Message):
    """List of consent scopes that are applicable to the explained access on a
  given resource.

  Fields:
    consentScopes: List of applicable consent scopes. Sorted in order of actor
      such that scopes belonging to the same actor will be adjacent to each
      other in the list.
    warning: Warnings associated with this response. It inform user with
      exceeded scope limit errors.
  """
    consentScopes = _messages.MessageField('ExplainDataAccessConsentScope', 1, repeated=True)
    warning = _messages.StringField(2)