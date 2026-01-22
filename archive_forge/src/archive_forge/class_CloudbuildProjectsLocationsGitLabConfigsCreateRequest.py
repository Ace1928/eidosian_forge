from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsGitLabConfigsCreateRequest(_messages.Message):
    """A CloudbuildProjectsLocationsGitLabConfigsCreateRequest object.

  Fields:
    gitLabConfig: A GitLabConfig resource to be passed as the request body.
    gitlabConfigId: Optional. The ID to use for the GitLabConfig, which will
      become the final component of the GitLabConfig's resource name.
      gitlab_config_id must meet the following requirements: + They must
      contain only alphanumeric characters and dashes. + They can be 1-64
      characters long. + They must begin and end with an alphanumeric
      character
    parent: Required. Name of the parent resource.
  """
    gitLabConfig = _messages.MessageField('GitLabConfig', 1)
    gitlabConfigId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)