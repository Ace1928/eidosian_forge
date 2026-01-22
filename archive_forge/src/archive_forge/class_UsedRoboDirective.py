from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UsedRoboDirective(_messages.Message):
    """Additional details of a used Robo directive.

  Fields:
    resourceName: The name of the resource that was used.
  """
    resourceName = _messages.StringField(1)