from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EcKeyType(_messages.Message):
    """Describes an Elliptic Curve key that may be used in a Certificate issued
  from a CaPool.

  Enums:
    SignatureAlgorithmValueValuesEnum: Optional. A signature algorithm that
      must be used. If this is omitted, any EC-based signature algorithm will
      be allowed.

  Fields:
    signatureAlgorithm: Optional. A signature algorithm that must be used. If
      this is omitted, any EC-based signature algorithm will be allowed.
  """

    class SignatureAlgorithmValueValuesEnum(_messages.Enum):
        """Optional. A signature algorithm that must be used. If this is omitted,
    any EC-based signature algorithm will be allowed.

    Values:
      EC_SIGNATURE_ALGORITHM_UNSPECIFIED: Not specified. Signifies that any
        signature algorithm may be used.
      ECDSA_P256: Refers to the Elliptic Curve Digital Signature Algorithm
        over the NIST P-256 curve.
      ECDSA_P384: Refers to the Elliptic Curve Digital Signature Algorithm
        over the NIST P-384 curve.
      EDDSA_25519: Refers to the Edwards-curve Digital Signature Algorithm
        over curve 25519, as described in RFC 8410.
    """
        EC_SIGNATURE_ALGORITHM_UNSPECIFIED = 0
        ECDSA_P256 = 1
        ECDSA_P384 = 2
        EDDSA_25519 = 3
    signatureAlgorithm = _messages.EnumField('SignatureAlgorithmValueValuesEnum', 1)