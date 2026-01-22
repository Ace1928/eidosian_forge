from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UIElementTooDeep(_messages.Message):
    """A warning that the screen hierarchy is deeper than the recommended
  threshold.

  Fields:
    depth: The depth of the screen element
    screenId: The screen id of the element
    screenStateId: The screen state id of the element
  """
    depth = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    screenId = _messages.StringField(2)
    screenStateId = _messages.StringField(3)