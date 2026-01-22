from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BlankScreen(_messages.Message):
    """A warning that Robo encountered a screen that was mostly blank; this may
  indicate a problem with the app.

  Fields:
    screenId: The screen id of the element
  """
    screenId = _messages.StringField(1)