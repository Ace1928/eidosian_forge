from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListNetworkPoliciesResponse(_messages.Message):
    """Response message for VmwareEngine.ListNetworkPolicies

  Fields:
    networkPolicies: A list of network policies.
    nextPageToken: A token, which can be send as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    unreachable: Locations that could not be reached when making an aggregated
      query using wildcards.
  """
    networkPolicies = _messages.MessageField('NetworkPolicy', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)