from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivatecaProjectsLocationsCaPoolsCertificatesRevokeRequest(_messages.Message):
    """A PrivatecaProjectsLocationsCaPoolsCertificatesRevokeRequest object.

  Fields:
    name: Required. The resource name for this Certificate in the format
      `projects/*/locations/*/caPools/*/certificates/*`.
    revokeCertificateRequest: A RevokeCertificateRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    revokeCertificateRequest = _messages.MessageField('RevokeCertificateRequest', 2)