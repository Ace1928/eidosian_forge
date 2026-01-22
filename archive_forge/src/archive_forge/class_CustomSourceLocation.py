from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomSourceLocation(_messages.Message):
    """Identifies the location of a custom souce.

  Fields:
    stateful: Whether this source is stateful.
  """
    stateful = _messages.BooleanField(1)