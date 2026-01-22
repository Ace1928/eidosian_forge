from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HorizontalPodAutoscaling(_messages.Message):
    """Configuration options for the horizontal pod autoscaling feature, which
  increases or decreases the number of replica pods a replication controller
  has based on the resource usage of the existing pods.

  Fields:
    disabled: Whether the Horizontal Pod Autoscaling feature is enabled in the
      cluster. When enabled, it ensures that metrics are collected into
      Stackdriver Monitoring.
  """
    disabled = _messages.BooleanField(1)