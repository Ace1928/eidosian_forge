from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesMigratingVmsReplicationCyclesListRequest(_messages.Message):
    """A
  VmmigrationProjectsLocationsSourcesMigratingVmsReplicationCyclesListRequest
  object.

  Fields:
    filter: Optional. The filter request.
    orderBy: Optional. the order by fields for the result.
    pageSize: Optional. The maximum number of replication cycles to return.
      The service may return fewer than this value. If unspecified, at most
      100 migrating VMs will be returned. The maximum value is 100; values
      above 100 will be coerced to 100.
    pageToken: Required. A page token, received from a previous
      `ListReplicationCycles` call. Provide this to retrieve the subsequent
      page. When paginating, all other parameters provided to
      `ListReplicationCycles` must match the call that provided the page
      token.
    parent: Required. The parent, which owns this collection of
      ReplicationCycles.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)