from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RayMonitoringConfig(_messages.Message):
    """RayMonitoringConfig specifies configuration of Ray Monitoring feature.

  Fields:
    enabled: When Ray addon is enabled in a cluster, this flag controls
      whether monitroing is enabled for Ray.
  """
    enabled = _messages.BooleanField(1)