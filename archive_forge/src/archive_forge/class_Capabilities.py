from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Capabilities(_messages.Message):
    """Capabilities adds and removes POSIX capabilities from running
  containers.

  Fields:
    add: Optional. Added capabilities +optional
    drop: Optional. Removed capabilities +optional
  """
    add = _messages.StringField(1, repeated=True)
    drop = _messages.StringField(2, repeated=True)