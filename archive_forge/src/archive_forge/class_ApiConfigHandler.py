from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApiConfigHandler(_messages.Message):
    """Google Cloud Endpoints (https://cloud.google.com/endpoints)
  configuration for API handlers.

  Enums:
    AuthFailActionValueValuesEnum: Action to take when users access resources
      that require authentication. Defaults to redirect.
    LoginValueValuesEnum: Level of login required to access this resource.
      Defaults to optional.
    SecurityLevelValueValuesEnum: Security (HTTPS) enforcement for this URL.

  Fields:
    authFailAction: Action to take when users access resources that require
      authentication. Defaults to redirect.
    login: Level of login required to access this resource. Defaults to
      optional.
    script: Path to the script from the application root directory.
    securityLevel: Security (HTTPS) enforcement for this URL.
    url: URL to serve the endpoint at.
  """

    class AuthFailActionValueValuesEnum(_messages.Enum):
        """Action to take when users access resources that require
    authentication. Defaults to redirect.

    Values:
      AUTH_FAIL_ACTION_UNSPECIFIED: Not specified. AUTH_FAIL_ACTION_REDIRECT
        is assumed.
      AUTH_FAIL_ACTION_REDIRECT: Redirects user to "accounts.google.com". The
        user is redirected back to the application URL after signing in or
        creating an account.
      AUTH_FAIL_ACTION_UNAUTHORIZED: Rejects request with a 401 HTTP status
        code and an error message.
    """
        AUTH_FAIL_ACTION_UNSPECIFIED = 0
        AUTH_FAIL_ACTION_REDIRECT = 1
        AUTH_FAIL_ACTION_UNAUTHORIZED = 2

    class LoginValueValuesEnum(_messages.Enum):
        """Level of login required to access this resource. Defaults to optional.

    Values:
      LOGIN_UNSPECIFIED: Not specified. LOGIN_OPTIONAL is assumed.
      LOGIN_OPTIONAL: Does not require that the user is signed in.
      LOGIN_ADMIN: If the user is not signed in, the auth_fail_action is
        taken. In addition, if the user is not an administrator for the
        application, they are given an error message regardless of
        auth_fail_action. If the user is an administrator, the handler
        proceeds.
      LOGIN_REQUIRED: If the user has signed in, the handler proceeds
        normally. Otherwise, the auth_fail_action is taken.
    """
        LOGIN_UNSPECIFIED = 0
        LOGIN_OPTIONAL = 1
        LOGIN_ADMIN = 2
        LOGIN_REQUIRED = 3

    class SecurityLevelValueValuesEnum(_messages.Enum):
        """Security (HTTPS) enforcement for this URL.

    Values:
      SECURE_UNSPECIFIED: Not specified.
      SECURE_DEFAULT: Both HTTP and HTTPS requests with URLs that match the
        handler succeed without redirects. The application can examine the
        request to determine which protocol was used, and respond accordingly.
      SECURE_NEVER: Requests for a URL that match this handler that use HTTPS
        are automatically redirected to the HTTP equivalent URL.
      SECURE_OPTIONAL: Both HTTP and HTTPS requests with URLs that match the
        handler succeed without redirects. The application can examine the
        request to determine which protocol was used and respond accordingly.
      SECURE_ALWAYS: Requests for a URL that match this handler that do not
        use HTTPS are automatically redirected to the HTTPS URL with the same
        path. Query parameters are reserved for the redirect.
    """
        SECURE_UNSPECIFIED = 0
        SECURE_DEFAULT = 1
        SECURE_NEVER = 2
        SECURE_OPTIONAL = 3
        SECURE_ALWAYS = 4
    authFailAction = _messages.EnumField('AuthFailActionValueValuesEnum', 1)
    login = _messages.EnumField('LoginValueValuesEnum', 2)
    script = _messages.StringField(3)
    securityLevel = _messages.EnumField('SecurityLevelValueValuesEnum', 4)
    url = _messages.StringField(5)