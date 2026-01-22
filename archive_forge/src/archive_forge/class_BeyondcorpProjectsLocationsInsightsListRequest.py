from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsInsightsListRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsInsightsListRequest object.

  Enums:
    ViewValueValuesEnum: Required. List only metadata or full data.

  Fields:
    filter: Optional. Filter expression to restrict the insights returned.
      Supported filter fields: * `type` * `category` * `subCategory` Examples:
      * "category = application AND type = count" * "category = application
      AND subCategory = iap" * "type = status" Allowed values: * type: [count,
      latency, status, list] * category: [application, device, request,
      security] * subCategory: [iap, webprotect] NOTE: Only equality based
      comparison is allowed. Only `AND` conjunction is allowed. NOTE: The
      'AND' in the filter field needs to be in capital letters only. NOTE:
      Just filtering on `subCategory` is not allowed. It should be passed in
      with the parent `category` too. (These expressions are based on the
      filter language described at https://google.aip.dev/160).
    orderBy: Optional. Hint for how to order the results. This is currently
      ignored.
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
      NOTE: Default page size is 50.
    pageToken: Optional. A token identifying a page of results the server
      should return.
    parent: Required. The resource name of InsightMetadata using the form:
      `organizations/{organization_id}/locations/{location}`
      `projects/{project_id}/locations/{location_id}`
    view: Required. List only metadata or full data.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Required. List only metadata or full data.

    Values:
      INSIGHT_VIEW_UNSPECIFIED: The default / unset value. The API will
        default to the BASIC view.
      BASIC: Include basic metadata about the insight, but not the insight
        data. This is the default value (for both ListInsights and
        GetInsight).
      FULL: Include everything.
    """
        INSIGHT_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 6)