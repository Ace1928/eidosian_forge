from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidationCA(_messages.Message):
    """Specification of ValidationCA. Defines the mechanism to obtain the
  Certificate Authority certificate to validate the peer certificate.

  Fields:
    certificateProviderInstance: The certificate provider instance
      specification that will be passed to the data plane, which will be used
      to load necessary credential information.
    grpcEndpoint: gRPC specific configuration to access the gRPC server to
      obtain the CA certificate.
  """
    certificateProviderInstance = _messages.MessageField('CertificateProviderInstance', 1)
    grpcEndpoint = _messages.MessageField('GoogleCloudNetworksecurityV1GrpcEndpoint', 2)