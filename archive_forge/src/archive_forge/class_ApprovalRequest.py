from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ApprovalRequest(_messages.Message):
    """A request for the customer to approve access to a resource.

  Fields:
    approve: Access was approved.
    dismiss: The request was dismissed.
    name: The resource name of the request. Format is "{projects|folders|organ
      izations}/{id}/approvalRequests/{approval_request}".
    requestTime: The time at which approval was requested.
    requestedDuration: The requested access duration.
    requestedExpiration: The original requested expiration for the approval.
      Calculated by adding the requested_duration to the request_time.
    requestedLocations: The locations for which approval is being requested.
    requestedReason: The justification for which approval is being requested.
    requestedResourceName: The resource for which approval is being requested.
      The format of the resource name is defined at
      https://cloud.google.com/apis/design/resource_names. The resource name
      here may either be a "full" resource name (e.g.
      "//library.googleapis.com/shelves/shelf1/books/book2") or a "relative"
      resource name (e.g. "shelves/shelf1/books/book2") as described in the
      resource name specification.
    requestedResourceProperties: Properties related to the resource
      represented by requested_resource_name.
  """
    approve = _messages.MessageField('ApproveDecision', 1)
    dismiss = _messages.MessageField('DismissDecision', 2)
    name = _messages.StringField(3)
    requestTime = _messages.StringField(4)
    requestedDuration = _messages.StringField(5)
    requestedExpiration = _messages.StringField(6)
    requestedLocations = _messages.MessageField('AccessLocations', 7)
    requestedReason = _messages.MessageField('AccessReason', 8)
    requestedResourceName = _messages.StringField(9)
    requestedResourceProperties = _messages.MessageField('ResourceProperties', 10)