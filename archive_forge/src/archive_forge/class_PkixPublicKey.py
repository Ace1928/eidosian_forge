from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PkixPublicKey(_messages.Message):
    """A public key in the PkixPublicKey format (see
  https://tools.ietf.org/html/rfc5280#section-4.1.2.7 for details). Public
  keys of this type are typically textually encoded using the PEM format.

  Enums:
    SignatureAlgorithmValueValuesEnum: The signature algorithm used to verify
      a message against a signature using this key. These signature algorithm
      must match the structure and any object identifiers encoded in
      `public_key_pem` (i.e. this algorithm must match that of the public
      key).

  Fields:
    publicKeyPem: A PEM-encoded public key, as described in
      https://tools.ietf.org/html/rfc7468#section-13
    signatureAlgorithm: The signature algorithm used to verify a message
      against a signature using this key. These signature algorithm must match
      the structure and any object identifiers encoded in `public_key_pem`
      (i.e. this algorithm must match that of the public key).
  """

    class SignatureAlgorithmValueValuesEnum(_messages.Enum):
        """The signature algorithm used to verify a message against a signature
    using this key. These signature algorithm must match the structure and any
    object identifiers encoded in `public_key_pem` (i.e. this algorithm must
    match that of the public key).

    Values:
      SIGNATURE_ALGORITHM_UNSPECIFIED: Not specified.
      RSA_PSS_2048_SHA256: RSASSA-PSS 2048 bit key with a SHA256 digest.
      RSA_SIGN_PSS_2048_SHA256: RSASSA-PSS 2048 bit key with a SHA256 digest.
      RSA_PSS_3072_SHA256: RSASSA-PSS 3072 bit key with a SHA256 digest.
      RSA_SIGN_PSS_3072_SHA256: RSASSA-PSS 3072 bit key with a SHA256 digest.
      RSA_PSS_4096_SHA256: RSASSA-PSS 4096 bit key with a SHA256 digest.
      RSA_SIGN_PSS_4096_SHA256: RSASSA-PSS 4096 bit key with a SHA256 digest.
      RSA_PSS_4096_SHA512: RSASSA-PSS 4096 bit key with a SHA512 digest.
      RSA_SIGN_PSS_4096_SHA512: RSASSA-PSS 4096 bit key with a SHA512 digest.
      RSA_SIGN_PKCS1_2048_SHA256: RSASSA-PKCS1-v1_5 with a 2048 bit key and a
        SHA256 digest.
      RSA_SIGN_PKCS1_3072_SHA256: RSASSA-PKCS1-v1_5 with a 3072 bit key and a
        SHA256 digest.
      RSA_SIGN_PKCS1_4096_SHA256: RSASSA-PKCS1-v1_5 with a 4096 bit key and a
        SHA256 digest.
      RSA_SIGN_PKCS1_4096_SHA512: RSASSA-PKCS1-v1_5 with a 4096 bit key and a
        SHA512 digest.
      ECDSA_P256_SHA256: ECDSA on the NIST P-256 curve with a SHA256 digest.
      EC_SIGN_P256_SHA256: ECDSA on the NIST P-256 curve with a SHA256 digest.
      ECDSA_P384_SHA384: ECDSA on the NIST P-384 curve with a SHA384 digest.
      EC_SIGN_P384_SHA384: ECDSA on the NIST P-384 curve with a SHA384 digest.
      ECDSA_P521_SHA512: ECDSA on the NIST P-521 curve with a SHA512 digest.
      EC_SIGN_P521_SHA512: ECDSA on the NIST P-521 curve with a SHA512 digest.
    """
        SIGNATURE_ALGORITHM_UNSPECIFIED = 0
        RSA_PSS_2048_SHA256 = 1
        RSA_SIGN_PSS_2048_SHA256 = 2
        RSA_PSS_3072_SHA256 = 3
        RSA_SIGN_PSS_3072_SHA256 = 4
        RSA_PSS_4096_SHA256 = 5
        RSA_SIGN_PSS_4096_SHA256 = 6
        RSA_PSS_4096_SHA512 = 7
        RSA_SIGN_PSS_4096_SHA512 = 8
        RSA_SIGN_PKCS1_2048_SHA256 = 9
        RSA_SIGN_PKCS1_3072_SHA256 = 10
        RSA_SIGN_PKCS1_4096_SHA256 = 11
        RSA_SIGN_PKCS1_4096_SHA512 = 12
        ECDSA_P256_SHA256 = 13
        EC_SIGN_P256_SHA256 = 14
        ECDSA_P384_SHA384 = 15
        EC_SIGN_P384_SHA384 = 16
        ECDSA_P521_SHA512 = 17
        EC_SIGN_P521_SHA512 = 18
    publicKeyPem = _messages.StringField(1)
    signatureAlgorithm = _messages.EnumField('SignatureAlgorithmValueValuesEnum', 2)