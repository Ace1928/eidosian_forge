from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1beta1ListWorkloadsResponse(_messages.Message):
    """Response of ListWorkloads endpoint.

  Fields:
    nextPageToken: The next page token. Return empty if reached the last page.
    workloads: List of Workloads under a given parent.
  """
    nextPageToken = _messages.StringField(1)
    workloads = _messages.MessageField('GoogleCloudAssuredworkloadsV1beta1Workload', 2, repeated=True)