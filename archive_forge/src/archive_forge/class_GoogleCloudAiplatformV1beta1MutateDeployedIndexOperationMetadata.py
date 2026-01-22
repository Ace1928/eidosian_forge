from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MutateDeployedIndexOperationMetadata(_messages.Message):
    """Runtime operation information for
  IndexEndpointService.MutateDeployedIndex.

  Fields:
    deployedIndexId: The unique index id specified by user
    genericMetadata: The operation generic information.
  """
    deployedIndexId = _messages.StringField(1)
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1beta1GenericOperationMetadata', 2)