from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsGithubEnterpriseConfigsCreateRequest(_messages.Message):
    """A CloudbuildProjectsGithubEnterpriseConfigsCreateRequest object.

  Fields:
    gheConfigId: Optional. The ID to use for the GithubEnterpriseConfig, which
      will become the final component of the GithubEnterpriseConfig's resource
      name. ghe_config_id must meet the following requirements: + They must
      contain only alphanumeric characters and dashes. + They can be 1-64
      characters long. + They must begin and end with an alphanumeric
      character
    gitHubEnterpriseConfig: A GitHubEnterpriseConfig resource to be passed as
      the request body.
    parent: Name of the parent project. For example:
      projects/{$project_number} or projects/{$project_id}
    projectId: ID of the project.
  """
    gheConfigId = _messages.StringField(1)
    gitHubEnterpriseConfig = _messages.MessageField('GitHubEnterpriseConfig', 2)
    parent = _messages.StringField(3, required=True)
    projectId = _messages.StringField(4)