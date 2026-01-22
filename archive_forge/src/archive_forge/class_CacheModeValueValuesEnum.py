from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CacheModeValueValuesEnum(_messages.Enum):
    """Optional. Set the CacheMode used by this route. BYPASS_CACHE and
    USE_ORIGIN_HEADERS proxy the origin's headers. Other cache modes pass
    Cache-Control to the client. Use client_ttl to override what is sent to
    the client.

    Values:
      CACHE_MODE_UNSPECIFIED: Unspecified value. Defaults to
        `CACHE_ALL_STATIC`.
      CACHE_ALL_STATIC: Automatically cache static content, including common
        image formats, media (video and audio), and web assets (JavaScript and
        CSS). Requests and responses that are marked as uncacheable, as well
        as dynamic content (including HTML), aren't cached.
      USE_ORIGIN_HEADERS: Only cache responses with valid HTTP caching
        directives. Responses without these headers aren't cached at Google's
        edge, and require a full trip to the origin on every request,
        potentially impacting performance and increasing load on the origin
        server.
      FORCE_CACHE_ALL: Cache all content, ignoring any `private`, `no-store`
        or `no-cache` directives in Cache-Control response headers.
        **Warning:** this might result in caching private, per-user (user
        identifiable) content. Only enable this on routes where the
        EdgeCacheOrigin doesn't serve private or dynamic content, such as
        storage buckets.
      BYPASS_CACHE: Bypass all caching for requests that match routes with
        this CDNPolicy applied. Enabling this causes the edge cache to ignore
        all HTTP caching directives. All responses are fulfilled from the
        origin.
    """
    CACHE_MODE_UNSPECIFIED = 0
    CACHE_ALL_STATIC = 1
    USE_ORIGIN_HEADERS = 2
    FORCE_CACHE_ALL = 3
    BYPASS_CACHE = 4