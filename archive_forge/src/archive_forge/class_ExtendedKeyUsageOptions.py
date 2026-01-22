from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExtendedKeyUsageOptions(_messages.Message):
    """KeyUsage.ExtendedKeyUsageOptions has fields that correspond to certain
  common OIDs that could be specified as an extended key usage value.

  Fields:
    clientAuth: Corresponds to OID 1.3.6.1.5.5.7.3.2. Officially described as
      "TLS WWW client authentication", though regularly used for non-WWW TLS.
    codeSigning: Corresponds to OID 1.3.6.1.5.5.7.3.3. Officially described as
      "Signing of downloadable executable code client authentication".
    emailProtection: Corresponds to OID 1.3.6.1.5.5.7.3.4. Officially
      described as "Email protection".
    ocspSigning: Corresponds to OID 1.3.6.1.5.5.7.3.9. Officially described as
      "Signing OCSP responses".
    serverAuth: Corresponds to OID 1.3.6.1.5.5.7.3.1. Officially described as
      "TLS WWW server authentication", though regularly used for non-WWW TLS.
    timeStamping: Corresponds to OID 1.3.6.1.5.5.7.3.8. Officially described
      as "Binding the hash of an object to a time".
  """
    clientAuth = _messages.BooleanField(1)
    codeSigning = _messages.BooleanField(2)
    emailProtection = _messages.BooleanField(3)
    ocspSigning = _messages.BooleanField(4)
    serverAuth = _messages.BooleanField(5)
    timeStamping = _messages.BooleanField(6)