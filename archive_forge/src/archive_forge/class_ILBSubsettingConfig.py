from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ILBSubsettingConfig(_messages.Message):
    """ILBSubsettingConfig contains the desired config of L4 Internal
  LoadBalancer subsetting on this cluster.

  Fields:
    enabled: Enables l4 ILB subsetting for this cluster.
  """
    enabled = _messages.BooleanField(1)