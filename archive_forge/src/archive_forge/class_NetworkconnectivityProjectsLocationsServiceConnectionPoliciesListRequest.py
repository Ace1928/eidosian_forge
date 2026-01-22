from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkconnectivityProjectsLocationsServiceConnectionPoliciesListRequest(_messages.Message):
    """A
  NetworkconnectivityProjectsLocationsServiceConnectionPoliciesListRequest
  object.

  Fields:
    filter: A filter expression that filters the results listed in the
      response.
    orderBy: Sort the results by a certain order.
    pageSize: The maximum number of results per page that should be returned.
    pageToken: The page token.
    parent: Required. The parent resource's name. ex.
      projects/123/locations/us-east1
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)