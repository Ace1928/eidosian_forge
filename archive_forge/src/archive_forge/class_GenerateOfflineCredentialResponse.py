from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateOfflineCredentialResponse(_messages.Message):
    """An offline credential for a cluster.

  Fields:
    clientCertificate: Output only. Client certificate to authenticate to k8s
      api-server.
    clientKey: Output only. Client private key to authenticate to k8s api-
      server.
    expireTime: Output only. Timestamp at which this credential will expire.
    userId: Output only. Client's identity.
  """
    clientCertificate = _messages.StringField(1)
    clientKey = _messages.StringField(2)
    expireTime = _messages.StringField(3)
    userId = _messages.StringField(4)