from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentUpdate(_messages.Message):
    """A DeploymentUpdate object.

  Fields:
    description: Output only. An optional user-provided description of the
      deployment after the current update has been applied.
    labels: Map of One Platform labels; provided by the client when the
      resource is created or updated. Specifically: Label keys must be between
      1 and 63 characters long and must conform to the following regular
      expression: `[a-z]([-a-z0-9]*[a-z0-9])?` Label values must be between 0
      and 63 characters long and must conform to the regular expression
      `([a-z]([-a-z0-9]*[a-z0-9])?)?`.
    manifest: Output only. URL of the manifest representing the update
      configuration of this deployment.
  """
    description = _messages.StringField(1)
    labels = _messages.MessageField('DeploymentUpdateLabelEntry', 2, repeated=True)
    manifest = _messages.StringField(3)