from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UrlMap(_messages.Message):
    """URL pattern and description of how the URL should be handled. App Engine
  can handle URLs by executing application code or by serving static files
  uploaded with the version, such as images, CSS, or JavaScript.

  Enums:
    AuthFailActionValueValuesEnum: Action to take when users access resources
      that require authentication. Defaults to redirect.
    LoginValueValuesEnum: Level of login required to access this resource. Not
      supported for Node.js in the App Engine standard environment.
    RedirectHttpResponseCodeValueValuesEnum: 30x code to use when performing
      redirects for the secure field. Defaults to 302.
    SecurityLevelValueValuesEnum: Security (HTTPS) enforcement for this URL.

  Fields:
    apiEndpoint: Uses API Endpoints to handle requests.
    authFailAction: Action to take when users access resources that require
      authentication. Defaults to redirect.
    login: Level of login required to access this resource. Not supported for
      Node.js in the App Engine standard environment.
    redirectHttpResponseCode: 30x code to use when performing redirects for
      the secure field. Defaults to 302.
    script: Executes a script to handle the requests that match this URL
      pattern. Only the auto value is supported for Node.js in the App Engine
      standard environment, for example "script": "auto".
    securityLevel: Security (HTTPS) enforcement for this URL.
    staticFiles: Returns the contents of a file, such as an image, as the
      response.
    urlRegex: URL prefix. Uses regular expression syntax, which means regexp
      special characters must be escaped, but should not contain groupings.
      All URLs that begin with this prefix are handled by this handler, using
      the portion of the URL after the prefix as part of the file path.
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
        """Level of login required to access this resource. Not supported for
    Node.js in the App Engine standard environment.

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

    class RedirectHttpResponseCodeValueValuesEnum(_messages.Enum):
        """30x code to use when performing redirects for the secure field.
    Defaults to 302.

    Values:
      REDIRECT_HTTP_RESPONSE_CODE_UNSPECIFIED: Not specified. 302 is assumed.
      REDIRECT_HTTP_RESPONSE_CODE_301: 301 Moved Permanently code.
      REDIRECT_HTTP_RESPONSE_CODE_302: 302 Moved Temporarily code.
      REDIRECT_HTTP_RESPONSE_CODE_303: 303 See Other code.
      REDIRECT_HTTP_RESPONSE_CODE_307: 307 Temporary Redirect code.
    """
        REDIRECT_HTTP_RESPONSE_CODE_UNSPECIFIED = 0
        REDIRECT_HTTP_RESPONSE_CODE_301 = 1
        REDIRECT_HTTP_RESPONSE_CODE_302 = 2
        REDIRECT_HTTP_RESPONSE_CODE_303 = 3
        REDIRECT_HTTP_RESPONSE_CODE_307 = 4

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
    apiEndpoint = _messages.MessageField('ApiEndpointHandler', 1)
    authFailAction = _messages.EnumField('AuthFailActionValueValuesEnum', 2)
    login = _messages.EnumField('LoginValueValuesEnum', 3)
    redirectHttpResponseCode = _messages.EnumField('RedirectHttpResponseCodeValueValuesEnum', 4)
    script = _messages.MessageField('ScriptHandler', 5)
    securityLevel = _messages.EnumField('SecurityLevelValueValuesEnum', 6)
    staticFiles = _messages.MessageField('StaticFilesHandler', 7)
    urlRegex = _messages.StringField(8)