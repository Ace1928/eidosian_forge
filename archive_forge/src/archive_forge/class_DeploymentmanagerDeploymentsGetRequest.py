from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerDeploymentsGetRequest(_messages.Message):
    """A DeploymentmanagerDeploymentsGetRequest object.

  Fields:
    deployment: The name of the deployment for this request.
    project: The project ID for this request.
  """
    deployment = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)