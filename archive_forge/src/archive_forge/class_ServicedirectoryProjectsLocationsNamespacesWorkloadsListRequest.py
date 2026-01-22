from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicedirectoryProjectsLocationsNamespacesWorkloadsListRequest(_messages.Message):
    """A ServicedirectoryProjectsLocationsNamespacesWorkloadsListRequest
  object.

  Fields:
    filter: Optional. The filter to list results by. General `filter` string
      syntax: ` ()` * `` can be any field name on the Workload proto. For
      example: `name`, `create_time`, `annotations.`, or `components` * `` can
      be `<`, `>`, `<=`, `>=`, `!=`, `=`, `:`. Of which `:` means `HAS`, and
      is roughly the same as `=` * `` must be the same data type as field * ``
      can be `AND`, `OR`, `NOT` Examples of valid filters: *
      `annotations.owner` returns workloads that have an annotation with the
      key `owner`, this is the same as `annotations:owner` *
      `components://compute.googleapis.com/projects/1234/zones/us-
      east1-c/instances/mig1\\ returns workloads that contain the specified
      component. * `name>projects/my-project/locations/us-east1/namespaces/my-
      namespace/workloads/workload-c` returns workloads that have names that
      are alphabetically later than the string, so "workload-e" is returned
      but "workload-a" is not * `annotations.owner!=sd AND
      annotations.foo=bar` returns workloads that have `owner` in annotation
      key but value is not `sd` AND have key/value `foo=bar` *
      `doesnotexist.foo=bar` returns an empty list. Note that workload doesn't
      have a field called "doesnotexist". Since the filter does not match any
      workloads, it returns no results For more information about filtering,
      see [API Filtering](https://aip.dev/160).
    orderBy: Optional. The order to list results by. General `order_by` string
      syntax: ` () (,)` * `` allows values: `name`, `display_name`,
      `create_time`, `update_time` * `` ascending or descending order by ``.
      If this is left blank, `asc` is used Note that an empty `order_by`
      string results in default order, which is order by `name` in ascending
      order.
    pageSize: Optional. The maximum number of items to return. The default
      value is 100.
    pageToken: Optional. The next_page_token value returned from a previous
      List request, if any.
    parent: Required. The resource name of the namespace whose service
      workloads you'd like to list.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)