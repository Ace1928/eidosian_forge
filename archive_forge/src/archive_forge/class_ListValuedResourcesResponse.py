from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListValuedResourcesResponse(_messages.Message):
    """Response message for listing the valued resources for a given
  simulation.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results.
    totalSize: The estimated total number of results matching the query.
    valuedResources: The valued resources that the attack path simulation
      identified.
  """
    nextPageToken = _messages.StringField(1)
    totalSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    valuedResources = _messages.MessageField('ValuedResource', 3, repeated=True)