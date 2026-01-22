from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeNetworkPolicy(_messages.Message):
    """NodeNetworkPolicy specifies if node network policy feature is enabled.
  This feature is only supported with DatapathProvider=ADVANCED_DATAPATH.

  Fields:
    enabled: Whether node network policy is enabled.
  """
    enabled = _messages.BooleanField(1)