from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsAppsListRequest(_messages.Message):
    """A ApigeeOrganizationsAppsListRequest object.

  Fields:
    apiProduct: API product.
    apptype: Optional. 'apptype' is no longer available. Use a 'filter'
      instead.
    expand: Optional. Flag that specifies whether to return an expanded list
      of apps for the organization. Defaults to `false`.
    filter: Optional. The filter expression to be used to get the list of
      apps, where filtering can be done on developerEmail, apiProduct,
      consumerKey, status, appId, appName, appType and appGroup. Examples:
      "developerEmail=foo@bar.com", "appType=AppGroup", or "appType=Developer"
      "filter" is supported from ver 1.10.0 and above.
    ids: Optional. Comma-separated list of app IDs on which to filter.
    includeCred: Optional. Flag that specifies whether to include credentials
      in the response.
    keyStatus: Optional. Key status of the app. Valid values include
      `approved` or `revoked`. Defaults to `approved`.
    pageSize: Optional. Count of apps a single page can have in the response.
      If unspecified, at most 100 apps will be returned. The maximum value is
      100; values above 100 will be coerced to 100. "page_size" is supported
      from ver 1.10.0 and above.
    pageToken: Optional. The starting index record for listing the developers.
      "page_token" is supported from ver 1.10.0 and above.
    parent: Required. Resource path of the parent in the following format:
      `organizations/{org}`
    rows: Optional. Maximum number of app IDs to return. Defaults to 10000.
    startKey: Returns the list of apps starting from the specified app ID.
    status: Optional. Filter by the status of the app. Valid values are
      `approved` or `revoked`. Defaults to `approved`.
  """
    apiProduct = _messages.StringField(1)
    apptype = _messages.StringField(2)
    expand = _messages.BooleanField(3)
    filter = _messages.StringField(4)
    ids = _messages.StringField(5)
    includeCred = _messages.BooleanField(6)
    keyStatus = _messages.StringField(7)
    pageSize = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(9)
    parent = _messages.StringField(10, required=True)
    rows = _messages.IntegerField(11)
    startKey = _messages.StringField(12)
    status = _messages.StringField(13)