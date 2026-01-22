import warnings
from typing import Tuple
from urllib.parse import urlparse
from w3lib.url import safe_url_string
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.http import Request, Response
from scrapy.utils.misc import load_object
from scrapy.utils.python import to_unicode
from scrapy.utils.url import strip_url
class OriginWhenCrossOriginPolicy(ReferrerPolicy):
    """
    https://www.w3.org/TR/referrer-policy/#referrer-policy-origin-when-cross-origin

    The "origin-when-cross-origin" policy specifies that a full URL,
    stripped for use as a referrer, is sent as referrer information
    when making same-origin requests from a particular request client,
    and only the ASCII serialization of the origin of the request client
    is sent as referrer information when making cross-origin requests
    from a particular request client.
    """
    name: str = POLICY_ORIGIN_WHEN_CROSS_ORIGIN

    def referrer(self, response_url, request_url):
        origin = self.origin(response_url)
        if origin == self.origin(request_url):
            return self.stripped_referrer(response_url)
        return origin