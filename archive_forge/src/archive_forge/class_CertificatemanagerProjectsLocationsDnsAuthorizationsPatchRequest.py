from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsDnsAuthorizationsPatchRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsDnsAuthorizationsPatchRequest
  object.

  Fields:
    dnsAuthorization: A DnsAuthorization resource to be passed as the request
      body.
    name: A user-defined name of the dns authorization. DnsAuthorization names
      must be unique globally and match pattern
      `projects/*/locations/*/dnsAuthorizations/*`.
    updateMask: Required. The update mask applies to the resource. For the
      `FieldMask` definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask.
  """
    dnsAuthorization = _messages.MessageField('DnsAuthorization', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)