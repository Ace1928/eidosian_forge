from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListScopesResponse(_messages.Message):
    """List of Scopes.

  Fields:
    nextPageToken: A token to request the next page of resources from the
      `ListScopes` method. The value of an empty string means that there are
      no more resources to return.
    scopes: The list of Scopes
  """
    nextPageToken = _messages.StringField(1)
    scopes = _messages.MessageField('Scope', 2, repeated=True)