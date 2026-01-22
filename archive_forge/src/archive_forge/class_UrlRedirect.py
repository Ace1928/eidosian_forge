from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UrlRedirect(_messages.Message):
    """The HTTP redirect configuration for a given request.

  Enums:
    RedirectResponseCodeValueValuesEnum: Optional. The HTTP status code to use
      for this redirect action. For a list of supported values, see
      RedirectResponseCode.

  Fields:
    hostRedirect: Optional. The host that is used in the redirect response
      instead of the one that was supplied in the request. The value must be
      between 1 and 255 characters.
    httpsRedirect: Optional. Determines whether the URL scheme in the
      redirected request is adjusted to `HTTPS` or remains that of the
      request. If it is set to `true` and at least one edge_ssl_certificates
      is set on the service, the URL scheme in the redirected request is set
      to `HTTPS`. If it is set to `false`, the URL scheme of the redirected
      request remains the same as that of the request.
    pathRedirect: Optional. The path that is used in the redirect response
      instead of the one that was supplied in the request. `path_redirect`
      cannot be supplied together with prefix_redirect. Supply one alone or
      neither. If neither is supplied, the path of the original request is
      used for the redirect. The path value must be between 1 and 1024
      characters.
    prefixRedirect: Optional. The prefix that replaces the prefix_match
      specified in the RouteRule rule, retaining the remaining portion of the
      URL before redirecting the request. `prefix_redirect` cannot be supplied
      together with path_redirect. Supply one alone or neither. If neither is
      supplied, the path of the original request is used for the redirect. The
      prefix value must be between 1 and 1024 characters.
    redirectResponseCode: Optional. The HTTP status code to use for this
      redirect action. For a list of supported values, see
      RedirectResponseCode.
    stripQuery: Optional. Determines whether accompanying query portions of
      the original URL are removed prior to redirecting the request. If it is
      set to `true`, the accompanying query portion of the original URL is
      removed before redirecting the request. If it is set to `false`, the
      query portion of the original URL is retained. The default is `false`.
  """

    class RedirectResponseCodeValueValuesEnum(_messages.Enum):
        """Optional. The HTTP status code to use for this redirect action. For a
    list of supported values, see RedirectResponseCode.

    Values:
      MOVED_PERMANENTLY_DEFAULT: `HTTP 301 (Moved Permanently)`
      FOUND: HTTP 302 Found
      SEE_OTHER: HTTP 303 See Other
      TEMPORARY_REDIRECT: `HTTP 307 (Temporary Redirect)`. In this case, the
        request method is retained.
      PERMANENT_REDIRECT: `HTTP 308 (Permanent Redirect)`. In this case, the
        request method is retained.
    """
        MOVED_PERMANENTLY_DEFAULT = 0
        FOUND = 1
        SEE_OTHER = 2
        TEMPORARY_REDIRECT = 3
        PERMANENT_REDIRECT = 4
    hostRedirect = _messages.StringField(1)
    httpsRedirect = _messages.BooleanField(2)
    pathRedirect = _messages.StringField(3)
    prefixRedirect = _messages.StringField(4)
    redirectResponseCode = _messages.EnumField('RedirectResponseCodeValueValuesEnum', 5)
    stripQuery = _messages.BooleanField(6)