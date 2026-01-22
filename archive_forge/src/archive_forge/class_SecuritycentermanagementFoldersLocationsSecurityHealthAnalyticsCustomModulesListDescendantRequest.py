from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycentermanagementFoldersLocationsSecurityHealthAnalyticsCustomModulesListDescendantRequest(_messages.Message):
    """A SecuritycentermanagementFoldersLocationsSecurityHealthAnalyticsCustomM
  odulesListDescendantRequest object.

  Fields:
    pageSize: Optional. The maximum number of results to return in a single
      response. Default is 10, minimum is 1, maximum is 1000.
    pageToken: Optional. A token identifying a page of results the server
      should return.
    parent: Required. Name of parent to list custom modules. Its format is
      "organizations/{organization}/locations/{location}",
      "folders/{folder}/locations/{location}", or
      "projects/{project}/locations/{location}"
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)