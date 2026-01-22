from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RetryBuildRequest(_messages.Message):
    """Specifies a build to retry.

  Fields:
    id: Required. Build ID of the original build.
    name: The name of the `Build` to retry. Format:
      `projects/{project}/locations/{location}/builds/{build}`
    projectId: Required. ID of the project.
  """
    id = _messages.StringField(1)
    name = _messages.StringField(2)
    projectId = _messages.StringField(3)