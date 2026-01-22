from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesClustersBackupsListRequest(_messages.Message):
    """A BigtableadminProjectsInstancesClustersBackupsListRequest object.

  Fields:
    filter: A filter expression that filters backups listed in the response.
      The expression must specify the field name, a comparison operator, and
      the value that you want to use for filtering. The value must be a
      string, a number, or a boolean. The comparison operator must be <, >,
      <=, >=, !=, =, or :. Colon ':' represents a HAS operator which is
      roughly synonymous with equality. Filter rules are case insensitive. The
      fields eligible for filtering are: * `name` * `source_table` * `state` *
      `start_time` (and values are of the format YYYY-MM-DDTHH:MM:SSZ) *
      `end_time` (and values are of the format YYYY-MM-DDTHH:MM:SSZ) *
      `expire_time` (and values are of the format YYYY-MM-DDTHH:MM:SSZ) *
      `size_bytes` To filter on multiple expressions, provide each separate
      expression within parentheses. By default, each expression is an AND
      expression. However, you can include AND, OR, and NOT expressions
      explicitly. Some examples of using filters are: * `name:"exact"` --> The
      backup's name is the string "exact". * `name:howl` --> The backup's name
      contains the string "howl". * `source_table:prod` --> The source_table's
      name contains the string "prod". * `state:CREATING` --> The backup is
      pending creation. * `state:READY` --> The backup is fully created and
      ready for use. * `(name:howl) AND (start_time <
      \\"2018-03-28T14:50:00Z\\")` --> The backup name contains the string
      "howl" and start_time of the backup is before 2018-03-28T14:50:00Z. *
      `size_bytes > 10000000000` --> The backup's size is greater than 10GB
    orderBy: An expression for specifying the sort order of the results of the
      request. The string value should specify one or more fields in Backup.
      The full syntax is described at https://aip.dev/132#ordering. Fields
      supported are: * name * source_table * expire_time * start_time *
      end_time * size_bytes * state For example, "start_time". The default
      sorting order is ascending. To specify descending order for the field, a
      suffix " desc" should be appended to the field name. For example,
      "start_time desc". Redundant space characters in the syntax are
      insigificant. If order_by is empty, results will be sorted by
      `start_time` in descending order starting from the most recently created
      backup.
    pageSize: Number of backups to be returned in the response. If 0 or less,
      defaults to the server's maximum allowed page size.
    pageToken: If non-empty, `page_token` should contain a next_page_token
      from a previous ListBackupsResponse to the same `parent` and with the
      same `filter`.
    parent: Required. The cluster to list backups from. Values are of the form
      `projects/{project}/instances/{instance}/clusters/{cluster}`. Use
      `{cluster} = '-'` to list backups for all clusters in an instance, e.g.,
      `projects/{project}/instances/{instance}/clusters/-`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)