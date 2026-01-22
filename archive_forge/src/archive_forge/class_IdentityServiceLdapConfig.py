from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityServiceLdapConfig(_messages.Message):
    """Configuration for the LDAP Auth flow.

  Fields:
    group: Optional. Contains the properties for locating and authenticating
      groups in the directory.
    server: Required. Server settings for the external LDAP server.
    serviceAccount: Required. Contains the credentials of the service account
      which is authorized to perform the LDAP search in the directory. The
      credentials can be supplied by the combination of the DN and password or
      the client certificate.
    user: Required. Defines where users exist in the LDAP directory.
  """
    group = _messages.MessageField('IdentityServiceGroupConfig', 1)
    server = _messages.MessageField('IdentityServiceServerConfig', 2)
    serviceAccount = _messages.MessageField('IdentityServiceServiceAccountConfig', 3)
    user = _messages.MessageField('IdentityServiceUserConfig', 4)