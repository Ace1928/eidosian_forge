from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalClusterOperationsConfig(_messages.Message):
    """Specifies the bare metal user cluster's observability infrastructure.

  Fields:
    enableApplicationLogs: Whether collection of application logs/metrics
      should be enabled (in addition to system logs/metrics).
  """
    enableApplicationLogs = _messages.BooleanField(1)