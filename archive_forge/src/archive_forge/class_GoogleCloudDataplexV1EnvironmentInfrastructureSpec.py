from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1EnvironmentInfrastructureSpec(_messages.Message):
    """Configuration for the underlying infrastructure used to run workloads.

  Fields:
    compute: Optional. Compute resources needed for analyze interactive
      workloads.
    osImage: Required. Software Runtime Configuration for analyze interactive
      workloads.
  """
    compute = _messages.MessageField('GoogleCloudDataplexV1EnvironmentInfrastructureSpecComputeResources', 1)
    osImage = _messages.MessageField('GoogleCloudDataplexV1EnvironmentInfrastructureSpecOsImageRuntime', 2)