from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsReportsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsReportsCreateRequest object.

  Fields:
    googleCloudApigeeV1CustomReport: A GoogleCloudApigeeV1CustomReport
      resource to be passed as the request body.
    parent: Required. The parent organization name under which the Custom
      Report will be created. Must be of the form:
      `organizations/{organization_id}/reports`
  """
    googleCloudApigeeV1CustomReport = _messages.MessageField('GoogleCloudApigeeV1CustomReport', 1)
    parent = _messages.StringField(2, required=True)