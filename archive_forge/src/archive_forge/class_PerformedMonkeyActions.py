from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerformedMonkeyActions(_messages.Message):
    """A notification that Robo performed some monkey actions.

  Fields:
    totalActions: The total number of monkey actions performed during the
      crawl.
  """
    totalActions = _messages.IntegerField(1, variant=_messages.Variant.INT32)