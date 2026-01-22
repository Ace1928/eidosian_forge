from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemoteWeb3SignerTemplate(_messages.Message):
    """Configuration to use an external key signing service, such as the
  service endpoint. The external key signer is expected to be managed entirely
  by the customer. For reference see https://docs.web3signer.consensys.net/
  for details on Web3Signer and
  https://docs.web3signer.consensys.net/reference/api/json-rpc for the API
  exposed by the external service.

  Fields:
    rootCertificate: Optional. Immutable. PEM-format X.509 certificate
      corresponding to the URI of the Web3Signer. An example of this can be
      found on https://www.ssl.com/guide/pem-der-crt-and-cer-x-509-encodings-
      and-conversions/ When not set, the validator client will only accept TLS
      certificates signed by well known certificate authorities (as in, the
      set configured by default in the OS Docker image).
    timeoutDuration: Optional. Timeout for requests to the Web3Signer service.
    votingPublicKeys: Required. The public key of the validator, as a
      hexadecimal string prefixed with "0x". This is used as the identifier
      for the key when sending requests to the Web3Signer service.
    web3signerUri: Required. URI of the Web3Signer service the validator
      client connects to, to request signing of attestations, blocks, etc.
  """
    rootCertificate = _messages.StringField(1)
    timeoutDuration = _messages.StringField(2)
    votingPublicKeys = _messages.StringField(3, repeated=True)
    web3signerUri = _messages.StringField(4)