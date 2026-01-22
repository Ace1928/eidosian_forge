from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RayMetricSpec(_messages.Message):
    """Configuration for the Ray metrics.

  Fields:
    disabled: Optional. Flag to disable the Ray metrics collection.
  """
    disabled = _messages.BooleanField(1)