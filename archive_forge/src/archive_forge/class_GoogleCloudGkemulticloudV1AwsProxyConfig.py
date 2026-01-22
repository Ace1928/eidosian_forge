from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AwsProxyConfig(_messages.Message):
    """Details of a proxy config stored in AWS Secret Manager.

  Fields:
    secretArn: The ARN of the AWS Secret Manager secret that contains the
      HTTP(S) proxy configuration. The secret must be a JSON encoded proxy
      configuration as described in
      https://cloud.google.com/anthos/clusters/docs/multi-cloud/aws/how-
      to/use-a-proxy#create_a_proxy_configuration_file
    secretVersion: The version string of the AWS Secret Manager secret that
      contains the HTTP(S) proxy configuration.
  """
    secretArn = _messages.StringField(1)
    secretVersion = _messages.StringField(2)