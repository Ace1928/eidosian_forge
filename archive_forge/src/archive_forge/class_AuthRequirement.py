from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthRequirement(_messages.Message):
    """User-defined authentication requirements, including support for [JSON
  Web Token (JWT)](https://tools.ietf.org/html/draft-ietf-oauth-json-web-
  token-32).

  Fields:
    audiences: The list of JWT [audiences](https://tools.ietf.org/html/draft-
      ietf-oauth-json-web-token-32#section-4.1.3). that are allowed to access.
      A JWT containing any of these audiences will be accepted. When this
      setting is absent, only JWTs with audience
      "https://Service_name/API_name" will be accepted. For example, if no
      audiences are in the setting, LibraryService API will only accept JWTs
      with the following audience "https://library-
      example.googleapis.com/google.example.library.v1.LibraryService".
      Example:      audiences: bookstore_android.apps.googleusercontent.com,
      bookstore_web.apps.googleusercontent.com
    providerId: id from authentication provider.  Example:      provider_id:
      bookstore_auth
  """
    audiences = _messages.StringField(1)
    providerId = _messages.StringField(2)