from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServicecontrolV1HttpRequest(_messages.Message):
    """A common proto for logging HTTP requests. Only contains semantics
  defined by the HTTP specification. Product-specific logging information MUST
  be defined in a separate message.

  Fields:
    cacheFillBytes: The number of HTTP response bytes inserted into cache. Set
      only when a cache fill was attempted.
    cacheHit: Whether or not an entity was served from cache (with or without
      validation).
    cacheLookup: Whether or not a cache lookup was attempted.
    cacheValidatedWithOriginServer: Whether or not the response was validated
      with the origin server before being served from cache. This field is
      only meaningful if `cache_hit` is True.
    latency: The request processing latency on the server, from the time the
      request was received until the response was sent.
    protocol: Protocol used for the request. Examples: "HTTP/1.1", "HTTP/2",
      "websocket"
    referer: The referer URL of the request, as defined in [HTTP/1.1 Header
      Field
      Definitions](http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html).
    remoteIp: The IP address (IPv4 or IPv6) of the client that issued the HTTP
      request. Examples: `"192.168.1.1"`, `"FE80::0202:B3FF:FE1E:8329"`.
    requestMethod: The request method. Examples: `"GET"`, `"HEAD"`, `"PUT"`,
      `"POST"`.
    requestSize: The size of the HTTP request message in bytes, including the
      request headers and the request body.
    requestUrl: The scheme (http, https), the host name, the path, and the
      query portion of the URL that was requested. Example:
      `"http://example.com/some/info?color=red"`.
    responseSize: The size of the HTTP response message sent back to the
      client, in bytes, including the response headers and the response body.
    serverIp: The IP address (IPv4 or IPv6) of the origin server that the
      request was sent to.
    status: The response code indicating the status of the response. Examples:
      200, 404.
    userAgent: The user agent sent by the client. Example: `"Mozilla/4.0
      (compatible; MSIE 6.0; Windows 98; Q312461; .NET CLR 1.0.3705)"`.
  """
    cacheFillBytes = _messages.IntegerField(1)
    cacheHit = _messages.BooleanField(2)
    cacheLookup = _messages.BooleanField(3)
    cacheValidatedWithOriginServer = _messages.BooleanField(4)
    latency = _messages.StringField(5)
    protocol = _messages.StringField(6)
    referer = _messages.StringField(7)
    remoteIp = _messages.StringField(8)
    requestMethod = _messages.StringField(9)
    requestSize = _messages.IntegerField(10)
    requestUrl = _messages.StringField(11)
    responseSize = _messages.IntegerField(12)
    serverIp = _messages.StringField(13)
    status = _messages.IntegerField(14, variant=_messages.Variant.INT32)
    userAgent = _messages.StringField(15)