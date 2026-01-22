from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsCertificatesCreateRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsCertificatesCreateRequest object.

  Fields:
    certificate: A Certificate resource to be passed as the request body.
    certificateId: Required. A user-provided name of the certificate.
    parent: Required. The parent resource of the certificate. Must be in the
      format `projects/*/locations/*`.
  """
    certificate = _messages.MessageField('Certificate', 1)
    certificateId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)