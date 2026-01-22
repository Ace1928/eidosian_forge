from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsClientTlsPoliciesCreateRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsClientTlsPoliciesCreateRequest object.

  Fields:
    clientTlsPolicy: A ClientTlsPolicy resource to be passed as the request
      body.
    clientTlsPolicyId: Required. Short name of the ClientTlsPolicy resource to
      be created. This value should be 1-63 characters long, containing only
      letters, numbers, hyphens, and underscores, and should not start with a
      number. E.g. "client_mtls_policy".
    parent: Required. The parent resource of the ClientTlsPolicy. Must be in
      the format `projects/*/locations/{location}`.
  """
    clientTlsPolicy = _messages.MessageField('ClientTlsPolicy', 1)
    clientTlsPolicyId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)