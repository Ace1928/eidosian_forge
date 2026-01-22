from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HelpLink(_messages.Message):
    """Describes a URL link.

  Fields:
    description: Describes what the link offers.
    url: The URL of the link.
  """
    description = _messages.StringField(1)
    url = _messages.StringField(2)