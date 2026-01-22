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
class RefererMiddleware:

    def __init__(self, settings=None):
        self.default_policy = DefaultReferrerPolicy
        if settings is not None:
            self.default_policy = _load_policy_class(settings.get('REFERRER_POLICY'))

    @classmethod
    def from_crawler(cls, crawler):
        if not crawler.settings.getbool('REFERER_ENABLED'):
            raise NotConfigured
        mw = cls(crawler.settings)
        crawler.signals.connect(mw.request_scheduled, signal=signals.request_scheduled)
        return mw

    def policy(self, resp_or_url, request):
        """
        Determine Referrer-Policy to use from a parent Response (or URL),
        and a Request to be sent.

        - if a valid policy is set in Request meta, it is used.
        - if the policy is set in meta but is wrong (e.g. a typo error),
          the policy from settings is used
        - if the policy is not set in Request meta,
          but there is a Referrer-policy header in the parent response,
          it is used if valid
        - otherwise, the policy from settings is used.
        """
        policy_name = request.meta.get('referrer_policy')
        if policy_name is None:
            if isinstance(resp_or_url, Response):
                policy_header = resp_or_url.headers.get('Referrer-Policy')
                if policy_header is not None:
                    policy_name = to_unicode(policy_header.decode('latin1'))
        if policy_name is None:
            return self.default_policy()
        cls = _load_policy_class(policy_name, warning_only=True)
        return cls() if cls else self.default_policy()

    def process_spider_output(self, response, result, spider):
        return (self._set_referer(r, response) for r in result or ())

    async def process_spider_output_async(self, response, result, spider):
        async for r in result or ():
            yield self._set_referer(r, response)

    def _set_referer(self, r, response):
        if isinstance(r, Request):
            referrer = self.policy(response, r).referrer(response.url, r.url)
            if referrer is not None:
                r.headers.setdefault('Referer', referrer)
        return r

    def request_scheduled(self, request, spider):
        redirected_urls = request.meta.get('redirect_urls', [])
        if redirected_urls:
            request_referrer = request.headers.get('Referer')
            if request_referrer is not None:
                parent_url = safe_url_string(request_referrer)
                policy_referrer = self.policy(parent_url, request).referrer(parent_url, request.url)
                if policy_referrer != request_referrer:
                    if policy_referrer is None:
                        request.headers.pop('Referer')
                    else:
                        request.headers['Referer'] = policy_referrer