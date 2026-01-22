from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1UpdateTensorboardOperationMetadata(_messages.Message):
    """Details of operations that perform update Tensorboard.

  Fields:
    genericMetadata: Operation metadata for Tensorboard.
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1GenericOperationMetadata', 1)