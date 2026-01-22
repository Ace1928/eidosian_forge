from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsMembershipsListAdminRequest(_messages.Message):
    """A GkehubProjectsLocationsMembershipsListAdminRequest object.

  Fields:
    filter: Optional. Lists Memberships of admin clusters that match the
      filter expression.
    orderBy: Optional. One or more fields to compare and use to sort the
      output. See https://google.aip.dev/132#ordering.
    pageSize: Optional. When requesting a 'page' of resources, `page_size`
      specifies number of resources to return. If unspecified or set to 0, all
      resources will be returned.
    pageToken: Optional. Token returned by previous call to
      `ListAdminClusterMemberships` which specifies the position in the list
      from where to continue listing the resources.
    parent: Required. The parent (project and location) where the Memberships
      of admin cluster will be listed. Specified in the format
      `projects/*/locations/*`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)