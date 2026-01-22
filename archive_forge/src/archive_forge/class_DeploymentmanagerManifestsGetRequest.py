from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerManifestsGetRequest(_messages.Message):
    """A DeploymentmanagerManifestsGetRequest object.

  Fields:
    deployment: The name of the deployment for this request.
    manifest: The name of the manifest for this request.
    project: The project ID for this request.
  """
    deployment = _messages.StringField(1, required=True)
    manifest = _messages.StringField(2, required=True)
    project = _messages.StringField(3, required=True)