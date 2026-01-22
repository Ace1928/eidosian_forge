from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstancesSetMinCpuPlatformRequest(_messages.Message):
    """A InstancesSetMinCpuPlatformRequest object.

  Fields:
    minCpuPlatform: Minimum cpu/platform this instance should be started at.
  """
    minCpuPlatform = _messages.StringField(1)