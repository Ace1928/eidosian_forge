from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1OrganizationProjectMapping(_messages.Message):
    """A GoogleCloudApigeeV1OrganizationProjectMapping object.

  Fields:
    location: Output only. The Google Cloud region where control plane data is
      located. For more information, see
      https://cloud.google.com/about/locations/.
    organization: Name of the Apigee organization.
    projectId: Google Cloud project associated with the Apigee organization
    projectIds: DEPRECATED: Use `project_id`. An Apigee Organization is mapped
      to a single project.
  """
    location = _messages.StringField(1)
    organization = _messages.StringField(2)
    projectId = _messages.StringField(3)
    projectIds = _messages.StringField(4, repeated=True)