from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTransformersResponse(_messages.Message):
    """Response message for TransformersService.ListTransformers.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    transformers: The list of transformers.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    transformers = _messages.MessageField('Transformer', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)