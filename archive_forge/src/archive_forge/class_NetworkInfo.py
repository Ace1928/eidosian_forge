from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkInfo(_messages.Message):
    """For display only. Metadata associated with a Compute Engine network.

  Fields:
    displayName: Name of a Compute Engine network.
    matchedIpRange: The IP range that matches the test.
    uri: URI of a Compute Engine network.
  """
    displayName = _messages.StringField(1)
    matchedIpRange = _messages.StringField(2)
    uri = _messages.StringField(3)