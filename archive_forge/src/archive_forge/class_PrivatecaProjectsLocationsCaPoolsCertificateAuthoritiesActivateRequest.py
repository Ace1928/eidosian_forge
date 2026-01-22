from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesActivateRequest(_messages.Message):
    """A PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesActivateRequest
  object.

  Fields:
    activateCertificateAuthorityRequest: A ActivateCertificateAuthorityRequest
      resource to be passed as the request body.
    name: Required. The resource name for this CertificateAuthority in the
      format `projects/*/locations/*/caPools/*/certificateAuthorities/*`.
  """
    activateCertificateAuthorityRequest = _messages.MessageField('ActivateCertificateAuthorityRequest', 1)
    name = _messages.StringField(2, required=True)