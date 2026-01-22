from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritypostureOrganizationsLocationsReportsCreateIaCValidationReportRequest(_messages.Message):
    """A
  SecuritypostureOrganizationsLocationsReportsCreateIaCValidationReportRequest
  object.

  Fields:
    createIaCValidationReportRequest: A CreateIaCValidationReportRequest
      resource to be passed as the request body.
    parent: Required. The parent resource name. The format of this value is as
      follows: `organizations/{organization}/locations/{location}`
  """
    createIaCValidationReportRequest = _messages.MessageField('CreateIaCValidationReportRequest', 1)
    parent = _messages.StringField(2, required=True)