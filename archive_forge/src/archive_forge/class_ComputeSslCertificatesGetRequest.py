from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeSslCertificatesGetRequest(_messages.Message):
    """A ComputeSslCertificatesGetRequest object.

  Fields:
    project: Project ID for this request.
    sslCertificate: Name of the SslCertificate resource to return.
  """
    project = _messages.StringField(1, required=True)
    sslCertificate = _messages.StringField(2, required=True)