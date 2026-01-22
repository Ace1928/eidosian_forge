from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MasterAuth(_messages.Message):
    """The authentication information for accessing the master endpoint.
  Authentication can be done using HTTP basic auth or using client
  certificates.

  Fields:
    clientCertificate: [Output only] Base64-encoded public certificate used by
      clients to authenticate to the cluster endpoint.
    clientCertificateConfig: Configuration for client certificate
      authentication on the cluster. For clusters before v1.12, if no
      configuration is specified, a client certificate is issued.
    clientKey: [Output only] Base64-encoded private key used by clients to
      authenticate to the cluster endpoint.
    clusterCaCertificate: [Output only] Base64-encoded public certificate that
      is the root of trust for the cluster.
    password: The password to use for HTTP basic authentication to the master
      endpoint. Because the master endpoint is open to the Internet, you
      should create a strong password. If a password is provided for cluster
      creation, username must be non-empty. Warning: basic authentication is
      deprecated, and will be removed in GKE control plane versions 1.19 and
      newer. For a list of recommended authentication methods, see:
      https://cloud.google.com/kubernetes-engine/docs/how-to/api-server-
      authentication
    username: The username to use for HTTP basic authentication to the master
      endpoint. For clusters v1.6.0 and later, basic authentication can be
      disabled by leaving username unspecified (or setting it to the empty
      string). Warning: basic authentication is deprecated, and will be
      removed in GKE control plane versions 1.19 and newer. For a list of
      recommended authentication methods, see:
      https://cloud.google.com/kubernetes-engine/docs/how-to/api-server-
      authentication
  """
    clientCertificate = _messages.StringField(1)
    clientCertificateConfig = _messages.MessageField('ClientCertificateConfig', 2)
    clientKey = _messages.StringField(3)
    clusterCaCertificate = _messages.StringField(4)
    password = _messages.StringField(5)
    username = _messages.StringField(6)