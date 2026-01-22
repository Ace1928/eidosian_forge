from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcfsConfig(_messages.Message):
    """GcfsConfig contains configurations of Google Container File System
  (image streaming).

  Fields:
    enabled: Whether to use GCFS.
  """
    enabled = _messages.BooleanField(1)