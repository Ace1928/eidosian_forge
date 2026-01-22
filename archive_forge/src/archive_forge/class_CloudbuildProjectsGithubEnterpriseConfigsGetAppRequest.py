from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsGithubEnterpriseConfigsGetAppRequest(_messages.Message):
    """A CloudbuildProjectsGithubEnterpriseConfigsGetAppRequest object.

  Fields:
    enterpriseConfigResource: Required. The name of the enterprise config
      resource associated with the GitHub App. For example: "projects/{$projec
      t_id}/locations/{location_id}/githubEnterpriseConfigs/{$config_id}"
  """
    enterpriseConfigResource = _messages.StringField(1, required=True)