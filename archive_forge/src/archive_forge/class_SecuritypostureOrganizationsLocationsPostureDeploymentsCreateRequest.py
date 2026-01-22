from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritypostureOrganizationsLocationsPostureDeploymentsCreateRequest(_messages.Message):
    """A SecuritypostureOrganizationsLocationsPostureDeploymentsCreateRequest
  object.

  Fields:
    parent: Required. Value for parent. Format:
      organizations/{org_id}/locations/{location}
    postureDeployment: A PostureDeployment resource to be passed as the
      request body.
    postureDeploymentId: Required. User provided identifier. It should be
      unique in scope of an Organization and location.
  """
    parent = _messages.StringField(1, required=True)
    postureDeployment = _messages.MessageField('PostureDeployment', 2)
    postureDeploymentId = _messages.StringField(3)