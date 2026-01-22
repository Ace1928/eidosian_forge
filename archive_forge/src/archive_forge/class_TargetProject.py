from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetProject(_messages.Message):
    """TargetProject message represents a target Compute Engine project for a
  migration or a clone.

  Fields:
    createTime: Output only. The time this target project resource was created
      (not related to when the Compute Engine project it points to was
      created).
    description: The target project's description.
    name: Output only. The name of the target project.
    project: Required. The target project ID (number) or project name.
    updateTime: Output only. The last time the target project resource was
      updated.
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    name = _messages.StringField(3)
    project = _messages.StringField(4)
    updateTime = _messages.StringField(5)