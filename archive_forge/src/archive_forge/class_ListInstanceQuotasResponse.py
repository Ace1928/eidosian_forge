from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListInstanceQuotasResponse(_messages.Message):
    """Response message for the list of Instance provisioning quotas.

  Fields:
    instanceQuotas: The provisioning quotas registered in this project.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    instanceQuotas = _messages.MessageField('InstanceQuota', 1, repeated=True)
    nextPageToken = _messages.StringField(2)