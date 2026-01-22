from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsGithubEnterpriseConfigsDeleteRequest(_messages.Message):
    """A CloudbuildProjectsGithubEnterpriseConfigsDeleteRequest object.

  Fields:
    configId: Unique identifier of the `GitHubEnterpriseConfig`
    name: This field should contain the name of the enterprise config
      resource. For example: "projects/{$project_id}/locations/{$location_id}/
      githubEnterpriseConfigs/{$config_id}"
    projectId: ID of the project
  """
    configId = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3)