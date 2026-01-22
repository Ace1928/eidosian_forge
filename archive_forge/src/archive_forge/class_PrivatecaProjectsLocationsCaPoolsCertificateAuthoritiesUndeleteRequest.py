from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesUndeleteRequest(_messages.Message):
    """A PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesUndeleteRequest
  object.

  Fields:
    name: Required. The resource name for this CertificateAuthority in the
      format `projects/*/locations/*/caPools/*/certificateAuthorities/*`.
    undeleteCertificateAuthorityRequest: A UndeleteCertificateAuthorityRequest
      resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    undeleteCertificateAuthorityRequest = _messages.MessageField('UndeleteCertificateAuthorityRequest', 2)