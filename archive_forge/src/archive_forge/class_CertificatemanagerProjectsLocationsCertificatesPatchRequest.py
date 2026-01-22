from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsCertificatesPatchRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsCertificatesPatchRequest object.

  Fields:
    certificate: A Certificate resource to be passed as the request body.
    name: A user-defined name of the certificate. Certificate names must be
      unique globally and match pattern
      `projects/*/locations/*/certificates/*`.
    updateMask: Required. The update mask applies to the resource. For the
      `FieldMask` definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask.
  """
    certificate = _messages.MessageField('Certificate', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)