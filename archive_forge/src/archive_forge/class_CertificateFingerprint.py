from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateFingerprint(_messages.Message):
    """A group of fingerprints for the x509 certificate.

  Fields:
    sha256Hash: The SHA 256 hash, encoded in hexadecimal, of the DER x509
      certificate.
  """
    sha256Hash = _messages.StringField(1)