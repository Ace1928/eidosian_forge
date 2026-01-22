from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1CreateFeaturestoreOperationMetadata(_messages.Message):
    """Details of operations that perform create Featurestore.

  Fields:
    genericMetadata: Operation metadata for Featurestore.
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1beta1GenericOperationMetadata', 1)