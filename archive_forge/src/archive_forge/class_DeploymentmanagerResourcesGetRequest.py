from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerResourcesGetRequest(_messages.Message):
    """A DeploymentmanagerResourcesGetRequest object.

  Fields:
    deployment: The name of the deployment for this request.
    project: The project ID for this request.
    resource: The name of the resource for this request.
  """
    deployment = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    resource = _messages.StringField(3, required=True)