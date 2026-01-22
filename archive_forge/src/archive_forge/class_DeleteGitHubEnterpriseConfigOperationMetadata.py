from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeleteGitHubEnterpriseConfigOperationMetadata(_messages.Message):
    """Metadata for `DeleteGitHubEnterpriseConfig` operation.

  Fields:
    completeTime: Time the operation was completed.
    createTime: Time the operation was created.
    githubEnterpriseConfig: The resource name of the GitHubEnterprise to be
      deleted. Format:
      `projects/{project}/locations/{location}/githubEnterpriseConfigs/{id}`.
  """
    completeTime = _messages.StringField(1)
    createTime = _messages.StringField(2)
    githubEnterpriseConfig = _messages.StringField(3)