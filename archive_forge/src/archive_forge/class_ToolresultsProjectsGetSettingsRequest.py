from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsGetSettingsRequest(_messages.Message):
    """A ToolresultsProjectsGetSettingsRequest object.

  Fields:
    projectId: A Project id. Required.
  """
    projectId = _messages.StringField(1, required=True)