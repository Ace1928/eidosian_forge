from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FleetObservabilityFleetObservabilityLoggingState(_messages.Message):
    """Feature state for logging feature.

  Fields:
    defaultLog: The base feature state of fleet default log.
    scopeLog: The base feature state of fleet scope log.
  """
    defaultLog = _messages.MessageField('FleetObservabilityFleetObservabilityBaseFeatureState', 1)
    scopeLog = _messages.MessageField('FleetObservabilityFleetObservabilityBaseFeatureState', 2)