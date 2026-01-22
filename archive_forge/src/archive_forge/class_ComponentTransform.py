from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComponentTransform(_messages.Message):
    """Description of a transform executed as part of an execution stage.

  Fields:
    name: Dataflow service generated name for this source.
    originalTransform: User name for the original user transform with which
      this transform is most closely associated.
    userName: Human-readable name for this transform; may be user or system
      generated.
  """
    name = _messages.StringField(1)
    originalTransform = _messages.StringField(2)
    userName = _messages.StringField(3)