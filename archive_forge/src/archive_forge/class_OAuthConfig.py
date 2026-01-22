from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OAuthConfig(_messages.Message):
    """Looker instance OAuth login settings.

  Fields:
    clientId: Input only. Client ID from an external OAuth application. This
      is an input-only field, and thus will not be set in any responses.
    clientSecret: Input only. Client secret from an external OAuth
      application. This is an input-only field, and thus will not be set in
      any responses.
  """
    clientId = _messages.StringField(1)
    clientSecret = _messages.StringField(2)