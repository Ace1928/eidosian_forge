from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsCertificatesGetRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsCertificatesGetRequest object.

  Fields:
    name: Required. A name of the certificate to describe. Must be in the
      format `projects/*/locations/*/certificates/*`.
  """
    name = _messages.StringField(1, required=True)