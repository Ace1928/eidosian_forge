from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UsedRoboIgnoreDirective(_messages.Message):
    """Additional details of a used Robo directive with an ignore action. Note:
  This is a different scenario than unused directive.

  Fields:
    resourceName: The name of the resource that was ignored.
  """
    resourceName = _messages.StringField(1)