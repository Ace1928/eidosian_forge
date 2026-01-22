from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateProjectConfigRequest(_messages.Message):
    """Request for UpdateProjectConfig.

  Fields:
    projectConfig: The new configuration for the project.
    updateMask: A FieldMask specifying which fields of the project_config to
      modify. Only the fields in the mask will be modified. If no mask is
      provided, this request is no-op.
  """
    projectConfig = _messages.MessageField('ProjectConfig', 1)
    updateMask = _messages.StringField(2)