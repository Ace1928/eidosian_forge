from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsScopesListMembershipsRequest(_messages.Message):
    """A GkehubProjectsLocationsScopesListMembershipsRequest object.

  Fields:
    filter: Optional. Lists Memberships that match the filter expression,
      following the syntax outlined in https://google.aip.dev/160. Currently,
      filtering can be done only based on Memberships's `name`, `labels`,
      `create_time`, `update_time`, and `unique_id`.
    pageSize: Optional. When requesting a 'page' of resources, `page_size`
      specifies number of resources to return. If unspecified or set to 0, all
      resources will be returned. Pagination is currently not supported;
      therefore, setting this field does not have any impact for now.
    pageToken: Optional. Token returned by previous call to
      `ListBoundMemberships` which specifies the position in the list from
      where to continue listing the resources.
    scopeName: Required. Name of the Scope, in the format
      `projects/*/locations/global/scopes/*`, to which the Memberships are
      bound.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    scopeName = _messages.StringField(4, required=True)