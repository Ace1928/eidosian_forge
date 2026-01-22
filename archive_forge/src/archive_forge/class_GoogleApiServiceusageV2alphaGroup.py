from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServiceusageV2alphaGroup(_messages.Message):
    """Information about the group.

  Fields:
    description: The detailed description of the group.
    displayName: The display name of the group.
    name: The resource name of the group.
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    name = _messages.StringField(3)