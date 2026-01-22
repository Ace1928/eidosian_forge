from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsIndexEndpointsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsIndexEndpointsListRequest object.

  Fields:
    filter: Optional. An expression for filtering the results of the request.
      For field names both snake_case and camelCase are supported. *
      `index_endpoint` supports = and !=. `index_endpoint` represents the
      IndexEndpoint ID, ie. the last segment of the IndexEndpoint's
      resourcename. * `display_name` supports =, != and regex() (uses
      [re2](https://github.com/google/re2/wiki/Syntax) syntax) * `labels`
      supports general map functions that is: `labels.key=value` - key:value
      equality `labels.key:* or labels:key - key existence A key including a
      space must be quoted. `labels."a key"`. Some examples: *
      `index_endpoint="1"` * `display_name="myDisplayName"` *
      `regex(display_name, "^A") -> The display name starts with an A. *
      `labels.myKey="myValue"`
    pageSize: Optional. The standard list page size.
    pageToken: Optional. The standard list page token. Typically obtained via
      ListIndexEndpointsResponse.next_page_token of the previous
      IndexEndpointService.ListIndexEndpoints call.
    parent: Required. The resource name of the Location from which to list the
      IndexEndpoints. Format: `projects/{project}/locations/{location}`
    readMask: Optional. Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    readMask = _messages.StringField(5)