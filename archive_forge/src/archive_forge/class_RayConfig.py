from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RayConfig(_messages.Message):
    """Configuration options for the Ray add-on.

  Fields:
    enabled: Whether the Ray addon is enabled for this cluster.
  """
    enabled = _messages.BooleanField(1)