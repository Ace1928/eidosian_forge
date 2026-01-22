from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateIdentityConstraints(_messages.Message):
    """Describes constraints on a Certificate's Subject and SubjectAltNames.

  Fields:
    allowSubjectAltNamesPassthrough: Required. If this is true, the
      SubjectAltNames extension may be copied from a certificate request into
      the signed certificate. Otherwise, the requested SubjectAltNames will be
      discarded.
    allowSubjectPassthrough: Required. If this is true, the Subject field may
      be copied from a certificate request into the signed certificate.
      Otherwise, the requested Subject will be discarded.
    celExpression: Optional. A CEL expression that may be used to validate the
      resolved X.509 Subject and/or Subject Alternative Name before a
      certificate is signed. To see the full allowed syntax and some examples,
      see https://cloud.google.com/certificate-authority-service/docs/using-
      cel
  """
    allowSubjectAltNamesPassthrough = _messages.BooleanField(1)
    allowSubjectPassthrough = _messages.BooleanField(2)
    celExpression = _messages.MessageField('Expr', 3)