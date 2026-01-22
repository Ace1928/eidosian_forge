from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServerConfig(_messages.Message):
    """Server settings for the external LDAP server.

  Fields:
    certificateAuthorityData: Optional. Contains a Base64 encoded, PEM
      formatted certificate authority certificate for the LDAP server. This
      must be provided for the "ldaps" and "startTLS" connections.
    connectionType: Optional. Defines the connection type to communicate with
      the LDAP server. If `starttls` or `ldaps` is specified, the
      certificate_authority_data should not be empty.
    host: Required. Defines the hostname or IP of the LDAP server. Port is
      optional and will default to 389, if unspecified. For example,
      "ldap.server.example" or "10.10.10.10:389".
  """
    certificateAuthorityData = _messages.BytesField(1)
    connectionType = _messages.StringField(2)
    host = _messages.StringField(3)