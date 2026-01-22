from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsGithubInstallationsPatchRequest(_messages.Message):
    """A CloudbuildProjectsGithubInstallationsPatchRequest object.

  Fields:
    id: GitHub installation ID, created by GitHub.
    installation: A Installation resource to be passed as the request body.
    installationId: Unique identifier of the GitHub installation. Deprecated.
      Should set installation.id
    name: The name of the `GitHubInstallation` to update. Format:
      `projects/{project}/locations/{location}/installations/{installation}`
    projectId: The project ID of the GCP project the installation is
      associated with.
    projectId1: ID of the project.
    updateMask: Update mask for the Installation resource. If this is set, the
      server will only update the fields specified in the field mask.
      Otherwise, a full update of the resource will be performed.
  """
    id = _messages.IntegerField(1, required=True)
    installation = _messages.MessageField('Installation', 2)
    installationId = _messages.IntegerField(3)
    name = _messages.StringField(4)
    projectId = _messages.StringField(5, required=True)
    projectId1 = _messages.StringField(6)
    updateMask = _messages.StringField(7)