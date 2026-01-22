from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TensorFlowVersion(_messages.Message):
    """A tensorflow version that a Node can be configured with.

  Fields:
    name: The resource name.
    version: the tensorflow version.
  """
    name = _messages.StringField(1)
    version = _messages.StringField(2)