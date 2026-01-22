from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OverlappingUIElements(_messages.Message):
    """A warning that Robo encountered a screen that has overlapping clickable
  elements; this may indicate a potential UI issue.

  Fields:
    resourceName: Resource names of the overlapping screen elements
    screenId: The screen id of the elements
  """
    resourceName = _messages.StringField(1, repeated=True)
    screenId = _messages.StringField(2)