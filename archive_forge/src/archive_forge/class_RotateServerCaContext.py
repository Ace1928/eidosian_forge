from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RotateServerCaContext(_messages.Message):
    """Instance rotate server CA context.

  Fields:
    kind: This is always `sql#rotateServerCaContext`.
    nextVersion: The fingerprint of the next version to be rotated to. If left
      unspecified, will be rotated to the most recently added server CA
      version.
  """
    kind = _messages.StringField(1)
    nextVersion = _messages.StringField(2)