from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMulticastGroupDefinitionsListRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMulticastGroupDefinitionsListRequest
  object.

  Fields:
    filter: A filter expression that filters the resources listed in the
      response. The expression must be of the form ` ` where operators: `<`,
      `>`, `<=`, `>=`, `!=`, `=`, `:` are supported (colon `:` represents a
      HAS operator which is roughly synonymous with equality). can refer to a
      proto or JSON field, or a synthetic field. Field names can be camelCase
      or snake_case. Examples: * Filter by name: name = "RESOURCE_NAME" *
      Filter by labels: * Resources that have a key named `foo` labels.foo:* *
      Resources that have a key named `foo` whose value is `bar` labels.foo =
      bar
    orderBy: A field used to sort the results by a certain order.
    pageSize: The maximum number of multicast group definitions to return per
      call.
    pageToken: A page token from an earlier query, as returned in
      `next_page_token`.
    parent: Required. The parent resource for which to list multicast group
      definitions. Use the following format: `projects/*/locations/global`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)