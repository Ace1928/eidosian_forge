from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkTags(_messages.Message):
    """Collection of Compute Engine network tags that can be applied to a
  node's underlying VM instance.

  Fields:
    tags: List of network tags.
  """
    tags = _messages.StringField(1, repeated=True)