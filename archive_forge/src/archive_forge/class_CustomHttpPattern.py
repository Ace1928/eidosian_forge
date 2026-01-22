from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomHttpPattern(_messages.Message):
    """A custom pattern is used for defining custom HTTP verb.

  Fields:
    kind: The name of this custom HTTP verb.
    path: The path matched by this custom verb.
  """
    kind = _messages.StringField(1)
    path = _messages.StringField(2)