from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitLabConfig(_messages.Message):
    """GitLabConfig represents the configuration for a GitLab integration.

  Fields:
    connectedRepositories: Connected GitLab.com or GitLabEnterprise
      repositories for this config.
    createTime: Output only. Time when the config was created.
    enterpriseConfig: Optional. GitLabEnterprise config.
    name: The resource name for the config.
    secrets: Required. Secret Manager secrets needed by the config.
    username: Username of the GitLab.com or GitLab Enterprise account Cloud
      Build will use.
    webhookKey: Output only. UUID included in webhook requests. The UUID is
      used to look up the corresponding config.
  """
    connectedRepositories = _messages.MessageField('GitLabRepositoryId', 1, repeated=True)
    createTime = _messages.StringField(2)
    enterpriseConfig = _messages.MessageField('GitLabEnterpriseConfig', 3)
    name = _messages.StringField(4)
    secrets = _messages.MessageField('GitLabSecrets', 5)
    username = _messages.StringField(6)
    webhookKey = _messages.StringField(7)