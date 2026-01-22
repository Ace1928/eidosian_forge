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
class NoReferrerWhenDowngradePolicy(ReferrerPolicy):
    """
    https://www.w3.org/TR/referrer-policy/#referrer-policy-no-referrer-when-downgrade

    The "no-referrer-when-downgrade" policy sends a full URL along with requests
    from a TLS-protected environment settings object to a potentially trustworthy URL,
    and requests from clients which are not TLS-protected to any origin.

    Requests from TLS-protected clients to non-potentially trustworthy URLs,
    on the other hand, will contain no referrer information.
    A Referer HTTP header will not be sent.

    This is a user agent's default behavior, if no policy is otherwise specified.
    """
    name: str = POLICY_NO_REFERRER_WHEN_DOWNGRADE

    def referrer(self, response_url, request_url):
        if not self.tls_protected(response_url) or self.tls_protected(request_url):
            return self.stripped_referrer(response_url)