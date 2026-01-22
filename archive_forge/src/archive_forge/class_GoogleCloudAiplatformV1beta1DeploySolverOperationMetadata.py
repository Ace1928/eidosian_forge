from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1DeploySolverOperationMetadata(_messages.Message):
    """Runtime operation information for SolverService.DeploySolver.

  Fields:
    genericMetadata: The generic operation information.
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1beta1GenericOperationMetadata', 1)