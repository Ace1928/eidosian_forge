from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateDescription(_messages.Message):
    """A CertificateDescription describes an X.509 certificate or CSR that has
  been issued, as an alternative to using ASN.1 / X.509.

  Fields:
    aiaIssuingCertificateUrls: Describes lists of issuer CA certificate URLs
      that appear in the "Authority Information Access" extension in the
      certificate.
    authorityKeyId: Identifies the subject_key_id of the parent certificate,
      per https://tools.ietf.org/html/rfc5280#section-4.2.1.1
    certFingerprint: The hash of the x.509 certificate.
    crlDistributionPoints: Describes a list of locations to obtain CRL
      information, i.e. the DistributionPoint.fullName described by
      https://tools.ietf.org/html/rfc5280#section-4.2.1.13
    publicKey: The public key that corresponds to an issued certificate.
    subjectDescription: Describes some of the values in a certificate that are
      related to the subject and lifetime.
    subjectKeyId: Provides a means of identifiying certificates that contain a
      particular public key, per
      https://tools.ietf.org/html/rfc5280#section-4.2.1.2.
    x509Description: Describes some of the technical X.509 fields in a
      certificate.
  """
    aiaIssuingCertificateUrls = _messages.StringField(1, repeated=True)
    authorityKeyId = _messages.MessageField('KeyId', 2)
    certFingerprint = _messages.MessageField('CertificateFingerprint', 3)
    crlDistributionPoints = _messages.StringField(4, repeated=True)
    publicKey = _messages.MessageField('PublicKey', 5)
    subjectDescription = _messages.MessageField('SubjectDescription', 6)
    subjectKeyId = _messages.MessageField('KeyId', 7)
    x509Description = _messages.MessageField('X509Parameters', 8)