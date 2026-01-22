from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VerticalPodAutoscaling(_messages.Message):
    """VerticalPodAutoscaling contains global, per-cluster information required
  by Vertical Pod Autoscaler to automatically adjust the resources of pods
  controlled by it.

  Fields:
    enabled: Enables vertical pod autoscaling.
  """
    enabled = _messages.BooleanField(1)