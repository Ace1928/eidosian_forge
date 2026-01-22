from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrafeasV1beta1DiscoveryDetails(_messages.Message):
    """Details of a discovery occurrence.

  Fields:
    discovered: Required. Analysis status for the discovered resource.
  """
    discovered = _messages.MessageField('Discovered', 1)