from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiagnoseNetworkResponse(_messages.Message):
    """DiagnoseNetworkResponse contains the current status for a specific
  network.

  Fields:
    result: The network status of a specific network.
    updateTime: The time when the network status was last updated.
  """
    result = _messages.MessageField('NetworkStatus', 1)
    updateTime = _messages.StringField(2)