from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudRouter(_messages.Message):
    """The Cloud Router info.

  Fields:
    name: The associated Cloud Router name.
  """
    name = _messages.StringField(1)