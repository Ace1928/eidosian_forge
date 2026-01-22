from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesRolloutsListRequest(_messages.Message):
    """A ServicemanagementServicesRolloutsListRequest object.

  Fields:
    filter: Required. Use `filter` to return subset of rollouts. The following
      filters are supported: -- By status. For example,
      `filter='status=SUCCESS'` -- By strategy. For example,
      `filter='strategy=TrafficPercentStrategy'`
    pageSize: The max number of items to include in the response list. Page
      size is 50 if not specified. Maximum value is 100.
    pageToken: The token of the page to retrieve.
    serviceName: Required. The name of the service. See the
      [overview](https://cloud.google.com/service-management/overview) for
      naming requirements. For example: `example.googleapis.com`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    serviceName = _messages.StringField(4, required=True)