from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SyntheticMonitorTarget(_messages.Message):
    """Describes a Synthetic Monitor to be invoked by Uptime.

  Fields:
    cloudFunctionV2: Target a Synthetic Monitor GCFv2 instance.
  """
    cloudFunctionV2 = _messages.MessageField('CloudFunctionV2Target', 1)