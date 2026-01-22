from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RayLoggingConfig(_messages.Message):
    """RayLoggingConfig specifies configuration of Ray logging.

  Fields:
    enabled: When Ray addon is enabled in a cluster, this flag controls
      whether logging is enabled for Ray.
  """
    enabled = _messages.BooleanField(1)