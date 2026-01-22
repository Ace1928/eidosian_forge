from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementOperationsListRequest(_messages.Message):
    """A ServicemanagementOperationsListRequest object.

  Fields:
    filter: A string for filtering Operations. The following filter fields are
      supported: * serviceName: Required. Only `=` operator is allowed. *
      startTime: The time this job was started, in ISO 8601 format. Allowed
      operators are `>=`, `>`, `<=`, and `<`. * status: Can be `done`,
      `in_progress`, or `failed`. Allowed operators are `=`, and `!=`. Filter
      expression supports conjunction (AND) and disjunction (OR) logical
      operators. However, the serviceName restriction must be at the top-level
      and can only be combined with other restrictions via the AND logical
      operator. Examples: * `serviceName={some-service}.googleapis.com` *
      `serviceName={some-service}.googleapis.com AND startTime>="2017-02-01"`
      * `serviceName={some-service}.googleapis.com AND status=done` *
      `serviceName={some-service}.googleapis.com AND (status=done OR
      startTime>="2017-02-01")`
    name: Not used.
    pageSize: The maximum number of operations to return. If unspecified,
      defaults to 50. The maximum value is 100.
    pageToken: The standard list page token.
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)