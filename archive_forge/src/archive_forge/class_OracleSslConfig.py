from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OracleSslConfig(_messages.Message):
    """Oracle SSL configuration information.

  Fields:
    caCertificate: Input only. PEM-encoded certificate of the CA that signed
      the source database server's certificate.
    caCertificateSet: Output only. Indicates whether the ca_certificate field
      has been set for this Connection-Profile.
  """
    caCertificate = _messages.StringField(1)
    caCertificateSet = _messages.BooleanField(2)