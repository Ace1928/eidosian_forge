from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SamlIdpConfig(_messages.Message):
    """SAML IDP (identity provider) configuration.

  Fields:
    changePasswordUri: The **Change Password URL** of the identity provider.
      Users will be sent to this URL when changing their passwords at
      `myaccount.google.com`. This takes precedence over the change password
      URL configured at customer-level. Must use `HTTPS`.
    entityId: Required. The SAML **Entity ID** of the identity provider.
    logoutRedirectUri: The **Logout Redirect URL** (sign-out page URL) of the
      identity provider. When a user clicks the sign-out link on a Google
      page, they will be redirected to this URL. This is a pure redirect with
      no attached SAML `LogoutRequest` i.e. SAML single logout is not
      supported. Must use `HTTPS`.
    singleSignOnServiceUri: Required. The `SingleSignOnService` endpoint
      location (sign-in page URL) of the identity provider. This is the URL
      where the `AuthnRequest` will be sent. Must use `HTTPS`. Assumed to
      accept the `HTTP-Redirect` binding.
  """
    changePasswordUri = _messages.StringField(1)
    entityId = _messages.StringField(2)
    logoutRedirectUri = _messages.StringField(3)
    singleSignOnServiceUri = _messages.StringField(4)