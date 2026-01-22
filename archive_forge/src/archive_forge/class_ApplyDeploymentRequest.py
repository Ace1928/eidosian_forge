from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplyDeploymentRequest(_messages.Message):
    """Request object for `ApplyDeployment`. The resources in given deployment
  gets applied to Orchestration Cluster. A new revision is created when a
  deployment is applied.
  """