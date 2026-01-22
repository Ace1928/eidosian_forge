from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListNetworkQuotasResponse(_messages.Message):
    """Response message for the list of Network provisioning quotas.

  Fields:
    networkQuotas: The provisioning quotas registered in this project.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    networkQuotas = _messages.MessageField('NetworkQuota', 1, repeated=True)
    nextPageToken = _messages.StringField(2)