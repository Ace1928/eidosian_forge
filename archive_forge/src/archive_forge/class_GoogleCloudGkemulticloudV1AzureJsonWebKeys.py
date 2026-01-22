from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureJsonWebKeys(_messages.Message):
    """AzureJsonWebKeys is a valid JSON Web Key Set as specififed in RFC 7517.

  Fields:
    keys: The public component of the keys used by the cluster to sign token
      requests.
  """
    keys = _messages.MessageField('GoogleCloudGkemulticloudV1Jwk', 1, repeated=True)