from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DisplayDevice(_messages.Message):
    """A set of Display Device options

  Fields:
    enableDisplay: Optional. Enables display for the Compute Engine VM
  """
    enableDisplay = _messages.BooleanField(1)