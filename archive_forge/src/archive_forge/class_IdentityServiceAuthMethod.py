from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityServiceAuthMethod(_messages.Message):
    """Configuration of an auth method for a member/cluster. Only one
  authentication method (e.g., OIDC and LDAP) can be set per AuthMethod.

  Fields:
    azureadConfig: AzureAD specific Configuration.
    googleConfig: GoogleConfig specific configuration.
    ldapConfig: LDAP specific configuration.
    name: Identifier for auth config.
    oidcConfig: OIDC specific configuration.
    proxy: Proxy server address to use for auth method.
    samlConfig: SAML specific configuration.
  """
    azureadConfig = _messages.MessageField('IdentityServiceAzureADConfig', 1)
    googleConfig = _messages.MessageField('IdentityServiceGoogleConfig', 2)
    ldapConfig = _messages.MessageField('IdentityServiceLdapConfig', 3)
    name = _messages.StringField(4)
    oidcConfig = _messages.MessageField('IdentityServiceOidcConfig', 5)
    proxy = _messages.StringField(6)
    samlConfig = _messages.MessageField('IdentityServiceSamlConfig', 7)