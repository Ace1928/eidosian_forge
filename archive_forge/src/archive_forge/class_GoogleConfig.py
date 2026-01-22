from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleConfig(_messages.Message):
    """Configuration for the Google Plugin Auth flow.

  Fields:
    disable: Disable automatic configuration of Google Plugin on supported
      platforms.
  """
    disable = _messages.BooleanField(1)