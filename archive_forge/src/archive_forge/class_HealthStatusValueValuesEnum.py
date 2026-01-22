from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthStatusValueValuesEnum(_messages.Enum):
    """Optional query parameter for showing the health status of each network
    endpoint. Valid options are SKIP or SHOW. If you don't specify this
    parameter, the health status of network endpoints will not be provided.

    Values:
      SHOW: Show the health status for each network endpoint. Impacts latency
        of the call.
      SKIP: Health status for network endpoints will not be provided.
    """
    SHOW = 0
    SKIP = 1