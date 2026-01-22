from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TypeProvidersListTypesResponse(_messages.Message):
    """A TypeProvidersListTypesResponse object.

  Fields:
    nextPageToken: A token used to continue a truncated list request.
    types: Output only. A list of resource type info.
  """
    nextPageToken = _messages.StringField(1)
    types = _messages.MessageField('TypeInfo', 2, repeated=True)