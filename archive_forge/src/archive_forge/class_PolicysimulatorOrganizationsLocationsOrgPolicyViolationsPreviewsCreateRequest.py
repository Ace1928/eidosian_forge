from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsCreateRequest(_messages.Message):
    """A PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsCreate
  Request object.

  Fields:
    googleCloudPolicysimulatorV1betaOrgPolicyViolationsPreview: A
      GoogleCloudPolicysimulatorV1betaOrgPolicyViolationsPreview resource to
      be passed as the request body.
    orgPolicyViolationsPreviewId: Optional. An optional user-specified ID for
      the OrgPolicyViolationsPreview. If not provided, a random ID will be
      generated.
    parent: Required. The organization under which this
      OrgPolicyViolationsPreview will be created. Example: `organizations/my-
      example-org/locations/global`
  """
    googleCloudPolicysimulatorV1betaOrgPolicyViolationsPreview = _messages.MessageField('GoogleCloudPolicysimulatorV1betaOrgPolicyViolationsPreview', 1)
    orgPolicyViolationsPreviewId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)