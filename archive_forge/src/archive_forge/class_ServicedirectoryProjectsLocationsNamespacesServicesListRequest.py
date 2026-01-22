from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicedirectoryProjectsLocationsNamespacesServicesListRequest(_messages.Message):
    """A ServicedirectoryProjectsLocationsNamespacesServicesListRequest object.

  Fields:
    filter: Optional. The filter to list results by. General `filter` string
      syntax: ` ()` * `` can be `name` or `annotations.` for map field * ``
      can be `<`, `>`, `<=`, `>=`, `!=`, `=`, `:`. Of which `:` means `HAS`,
      and is roughly the same as `=` * `` must be the same data type as field
      * `` can be `AND`, `OR`, `NOT` Examples of valid filters: *
      `annotations.owner` returns services that have a annotation with the key
      `owner`, this is the same as `annotations:owner` *
      `annotations.protocol=gRPC` returns services that have key/value
      `protocol=gRPC` * `name>projects/my-project/locations/us-
      east1/namespaces/my-namespace/services/service-c` returns services that
      have name that is alphabetically later than the string, so "service-e"
      is returned but "service-a" is not * `annotations.owner!=sd AND
      annotations.foo=bar` returns services that have `owner` in annotation
      key but value is not `sd` AND have key/value `foo=bar` *
      `doesnotexist.foo=bar` returns an empty list. Note that service doesn't
      have a field called "doesnotexist". Since the filter does not match any
      services, it returns no results For more information about filtering,
      see [API Filtering](https://aip.dev/160).
    orderBy: Optional. The order to list results by. General `order_by` string
      syntax: ` () (,)` * `` allows value: `name` * `` ascending or descending
      order by ``. If this is left blank, `asc` is used Note that an empty
      `order_by` string results in default order, which is order by `name` in
      ascending order.
    pageSize: Optional. The maximum number of items to return.
    pageToken: Optional. The next_page_token value returned from a previous
      List request, if any.
    parent: Required. The resource name of the namespace whose services you'd
      like to list.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)