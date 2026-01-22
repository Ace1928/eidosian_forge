from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreateGitLabConnectedRepositoriesResponseMetadata(_messages.Message):
    """Metadata for `BatchCreateGitLabConnectedRepositories` operation.

  Fields:
    completeTime: Time the operation was completed.
    config: The name of the `GitLabConfig` that added connected repositories.
      Format: `projects/{project}/locations/{location}/gitLabConfigs/{config}`
    createTime: Time the operation was created.
  """
    completeTime = _messages.StringField(1)
    config = _messages.StringField(2)
    createTime = _messages.StringField(3)