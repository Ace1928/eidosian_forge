from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesTablesAuthorizedViewsListRequest(_messages.Message):
    """A BigtableadminProjectsInstancesTablesAuthorizedViewsListRequest object.

  Enums:
    ViewValueValuesEnum: Optional. The resource_view to be applied to the
      returned views' fields. Default to NAME_ONLY.

  Fields:
    pageSize: Optional. Maximum number of results per page. A page_size of
      zero lets the server choose the number of items to return. A page_size
      which is strictly positive will return at most that many items. A
      negative page_size will cause an error. Following the first request,
      subsequent paginated calls are not required to pass a page_size. If a
      page_size is set in subsequent calls, it must match the page_size given
      in the first request.
    pageToken: Optional. The value of `next_page_token` returned by a previous
      call.
    parent: Required. The unique name of the table for which AuthorizedViews
      should be listed. Values are of the form
      `projects/{project}/instances/{instance}/tables/{table}`.
    view: Optional. The resource_view to be applied to the returned views'
      fields. Default to NAME_ONLY.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. The resource_view to be applied to the returned views'
    fields. Default to NAME_ONLY.

    Values:
      RESPONSE_VIEW_UNSPECIFIED: Uses the default view for each method as
        documented in the request.
      NAME_ONLY: Only populates `name`.
      BASIC: Only populates the AuthorizedView's basic metadata. This
        includes: name, deletion_protection, etag.
      FULL: Populates every fields.
    """
        RESPONSE_VIEW_UNSPECIFIED = 0
        NAME_ONLY = 1
        BASIC = 2
        FULL = 3
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)