from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsInstancesReportStatusRequest(_messages.Message):
    """A ApigeeOrganizationsInstancesReportStatusRequest object.

  Fields:
    googleCloudApigeeV1ReportInstanceStatusRequest: A
      GoogleCloudApigeeV1ReportInstanceStatusRequest resource to be passed as
      the request body.
    instance: The name of the instance reporting this status. For SaaS the
      request will be rejected if no instance exists under this name. Format
      is organizations/{org}/instances/{instance}
  """
    googleCloudApigeeV1ReportInstanceStatusRequest = _messages.MessageField('GoogleCloudApigeeV1ReportInstanceStatusRequest', 1)
    instance = _messages.StringField(2, required=True)