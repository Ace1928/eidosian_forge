from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessUrls(_messages.Message):
    """URLs where a CertificateAuthority will publish content.

  Fields:
    caCertificateAccessUrl: The URL where this CertificateAuthority's CA
      certificate is published. This will only be set for CAs that have been
      activated.
    crlAccessUrls: The URLs where this CertificateAuthority's CRLs are
      published. This will only be set for CAs that have been activated.
  """
    caCertificateAccessUrl = _messages.StringField(1)
    crlAccessUrls = _messages.StringField(2, repeated=True)