from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FeatureOnlineStoreDedicatedServingEndpoint(_messages.Message):
    """The dedicated serving endpoint for this FeatureOnlineStore. Only need to
  set when you choose Optimized storage type. Public endpoint is provisioned
  by default.

  Fields:
    publicEndpointDomainName: Output only. This field will be populated with
      the domain name to use for this FeatureOnlineStore
  """
    publicEndpointDomainName = _messages.StringField(1)