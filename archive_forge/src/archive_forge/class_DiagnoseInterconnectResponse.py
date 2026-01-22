from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiagnoseInterconnectResponse(_messages.Message):
    """DiagnoseInterconnectResponse contains the current diagnostics for a
  specific interconnect.

  Fields:
    result: The network status of a specific interconnect.
    updateTime: The time when the interconnect diagnostics was last updated.
  """
    result = _messages.MessageField('InterconnectDiagnostics', 1)
    updateTime = _messages.StringField(2)