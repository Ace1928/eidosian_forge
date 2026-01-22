from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsTrustConfigsPatchRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsTrustConfigsPatchRequest object.

  Fields:
    name: A user-defined name of the trust config. TrustConfig names must be
      unique globally and match pattern
      `projects/*/locations/*/trustConfigs/*`.
    trustConfig: A TrustConfig resource to be passed as the request body.
    updateMask: Required. The update mask applies to the resource. For the
      `FieldMask` definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask.
  """
    name = _messages.StringField(1, required=True)
    trustConfig = _messages.MessageField('TrustConfig', 2)
    updateMask = _messages.StringField(3)