from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListWorkloadIdentityPoolNamespacesResponse(_messages.Message):
    """Response message for ListWorkloadIdentityPoolNamespaces.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    workloadIdentityPoolNamespaces: A list of namespaces.
  """
    nextPageToken = _messages.StringField(1)
    workloadIdentityPoolNamespaces = _messages.MessageField('WorkloadIdentityPoolNamespace', 2, repeated=True)