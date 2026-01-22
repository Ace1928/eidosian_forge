from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPermittedScopesResponse(_messages.Message):
    """List of permitted Scopes.

  Fields:
    nextPageToken: A token to request the next page of resources from the
      `ListPermittedScopes` method. The value of an empty string means that
      there are no more resources to return.
    scopes: The list of permitted Scopes
  """
    nextPageToken = _messages.StringField(1)
    scopes = _messages.MessageField('Scope', 2, repeated=True)