from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRouteRedirect(_messages.Message):
    """The specification for redirecting traffic.

  Enums:
    ResponseCodeValueValuesEnum: The HTTP Status code to use for the redirect.

  Fields:
    hostRedirect: The host that will be used in the redirect response instead
      of the one that was supplied in the request.
    httpsRedirect: If set to true, the URL scheme in the redirected request is
      set to https. If set to false, the URL scheme of the redirected request
      will remain the same as that of the request. The default is set to
      false.
    pathRedirect: The path that will be used in the redirect response instead
      of the one that was supplied in the request. path_redirect can not be
      supplied together with prefix_redirect. Supply one alone or neither. If
      neither is supplied, the path of the original request will be used for
      the redirect.
    portRedirect: The port that will be used in the redirected request instead
      of the one that was supplied in the request.
    prefixRewrite: Indicates that during redirection, the matched prefix (or
      path) should be swapped with this value. This option allows URLs be
      dynamically created based on the request.
    responseCode: The HTTP Status code to use for the redirect.
    stripQuery: if set to true, any accompanying query portion of the original
      URL is removed prior to redirecting the request. If set to false, the
      query portion of the original URL is retained. The default is set to
      false.
  """

    class ResponseCodeValueValuesEnum(_messages.Enum):
        """The HTTP Status code to use for the redirect.

    Values:
      RESPONSE_CODE_UNSPECIFIED: Default value
      MOVED_PERMANENTLY_DEFAULT: Corresponds to 301.
      FOUND: Corresponds to 302.
      SEE_OTHER: Corresponds to 303.
      TEMPORARY_REDIRECT: Corresponds to 307. In this case, the request method
        will be retained.
      PERMANENT_REDIRECT: Corresponds to 308. In this case, the request method
        will be retained.
    """
        RESPONSE_CODE_UNSPECIFIED = 0
        MOVED_PERMANENTLY_DEFAULT = 1
        FOUND = 2
        SEE_OTHER = 3
        TEMPORARY_REDIRECT = 4
        PERMANENT_REDIRECT = 5
    hostRedirect = _messages.StringField(1)
    httpsRedirect = _messages.BooleanField(2)
    pathRedirect = _messages.StringField(3)
    portRedirect = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    prefixRewrite = _messages.StringField(5)
    responseCode = _messages.EnumField('ResponseCodeValueValuesEnum', 6)
    stripQuery = _messages.BooleanField(7)