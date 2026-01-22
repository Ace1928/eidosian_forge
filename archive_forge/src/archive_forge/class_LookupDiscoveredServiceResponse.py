from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookupDiscoveredServiceResponse(_messages.Message):
    """Response for LookupDiscoveredService.

  Fields:
    discoveredService: Discovered Service if exists, empty otherwise.
  """
    discoveredService = _messages.MessageField('DiscoveredService', 1)