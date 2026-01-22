from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsPatchDeploymentsResumeRequest(_messages.Message):
    """A OsconfigProjectsPatchDeploymentsResumeRequest object.

  Fields:
    name: Required. The resource name of the patch deployment in the form
      `projects/*/patchDeployments/*`.
    resumePatchDeploymentRequest: A ResumePatchDeploymentRequest resource to
      be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    resumePatchDeploymentRequest = _messages.MessageField('ResumePatchDeploymentRequest', 2)