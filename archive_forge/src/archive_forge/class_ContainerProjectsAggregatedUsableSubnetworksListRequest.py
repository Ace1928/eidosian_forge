from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerProjectsAggregatedUsableSubnetworksListRequest(_messages.Message):
    """A ContainerProjectsAggregatedUsableSubnetworksListRequest object.

  Fields:
    filter: Filtering currently only supports equality on the networkProjectId
      and must be in the form: "networkProjectId=[PROJECTID]", where
      `networkProjectId` is the project which owns the listed subnetworks.
      This defaults to the parent project ID.
    pageSize: The max number of results per page that should be returned. If
      the number of available results is larger than `page_size`, a
      `next_page_token` is returned which can be used to get the next page of
      results in subsequent requests. Acceptable values are 0 to 500,
      inclusive. (Default: 500)
    pageToken: Specifies a page token to use. Set this to the nextPageToken
      returned by previous list requests to get the next page of results.
    parent: The parent project where subnetworks are usable. Specified in the
      format `projects/*`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)