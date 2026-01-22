from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeatureOnlineStoreBigtable(_messages.Message):
    """A GoogleCloudAiplatformV1beta1FeatureOnlineStoreBigtable object.

  Fields:
    autoScaling: Required. Autoscaling config applied to Bigtable Instance.
  """
    autoScaling = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureOnlineStoreBigtableAutoScaling', 1)