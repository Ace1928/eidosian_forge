from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsListRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsListRequest object.

  Fields:
    filter: List filter.
    pageSize: The maximum number of cluster groups to return. The service may
      return fewer cluster groups.
    pageToken: A page token, received from a previous
      `ListClusterGroupsRequest` call. Provide this to retrieve the subsequent
      page. When paginating, you must provide exactly the same parameters to
      `ListClusterGroupsRequest` as you provided to the page token request.
    parent: Required. The location and project which will be queried for data
      centers. For example, projects/PROJECT-NUMBER/locations/us-central1
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)