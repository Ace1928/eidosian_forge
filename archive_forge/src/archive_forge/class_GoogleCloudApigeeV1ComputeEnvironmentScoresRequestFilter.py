from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ComputeEnvironmentScoresRequestFilter(_messages.Message):
    """Filter scores by component path. Used custom filter instead of AIP-160
  as the use cases are highly constrained and predictable.

  Fields:
    scorePath: Optional. Return scores for this component. Example:
      "/org@myorg/envgroup@myenvgroup/env@myenv/proxies/proxy@myproxy/source"
  """
    scorePath = _messages.StringField(1)