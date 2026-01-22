from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsUpdateProjectSettingsRequest(_messages.Message):
    """A ArtifactregistryProjectsUpdateProjectSettingsRequest object.

  Fields:
    name: The name of the project's settings. Always of the form:
      projects/{project-id}/projectSettings In update request: never set In
      response: always set
    projectSettings: A ProjectSettings resource to be passed as the request
      body.
    updateMask: Field mask to support partial updates.
  """
    name = _messages.StringField(1, required=True)
    projectSettings = _messages.MessageField('ProjectSettings', 2)
    updateMask = _messages.StringField(3)