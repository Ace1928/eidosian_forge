from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeployJobRunMetadata(_messages.Message):
    """DeployJobRunMetadata surfaces information associated with a
  `DeployJobRun` to the user.

  Fields:
    cloudRun: Output only. The name of the Cloud Run Service that is
      associated with a `DeployJobRun`.
    custom: Output only. Custom metadata provided by user-defined deploy
      operation.
    customTarget: Output only. Custom Target metadata associated with a
      `DeployJobRun`.
  """
    cloudRun = _messages.MessageField('CloudRunMetadata', 1)
    custom = _messages.MessageField('CustomMetadata', 2)
    customTarget = _messages.MessageField('CustomTargetDeployMetadata', 3)