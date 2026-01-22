from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsSecurityGatewaysListRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsSecurityGatewaysListRequest object.

  Fields:
    filter: Optional. A filter specifying constraints of a list operation. All
      fields in the SecurityGateway message are supported. For example, the
      following query will return the SecurityGateway with displayName "test-
      security-gateway" For more information, please refer to
      https://google.aip.dev/160.
    orderBy: Optional. Specifies the ordering of results. See [Sorting
      order](https://cloud.google.com/apis/design/design_patterns#sorting_orde
      r) for more information.
    pageSize: Optional. The maximum number of items to return. If not
      specified, a default value of 50 will be used by the service. Regardless
      of the page_size value, the response may include a partial list and a
      caller should only rely on response's next_page_token to determine if
      there are more instances left to be queried.
    pageToken: Optional. The next_page_token value returned from a previous
      ListSecurityGatewayRequest, if any.
    parent: Required. The parent location to which the resources belong.
      `projects/{project_id}/locations/{location_id}/`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)