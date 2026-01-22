import logging
from urllib.parse import urljoin, urlparse
from w3lib.url import safe_url_string
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import HtmlResponse
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.response import get_meta_refresh
class BaseRedirectMiddleware:
    enabled_setting = 'REDIRECT_ENABLED'

    def __init__(self, settings):
        if not settings.getbool(self.enabled_setting):
            raise NotConfigured
        self.max_redirect_times = settings.getint('REDIRECT_MAX_TIMES')
        self.priority_adjust = settings.getint('REDIRECT_PRIORITY_ADJUST')

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)

    def _redirect(self, redirected, request, spider, reason):
        ttl = request.meta.setdefault('redirect_ttl', self.max_redirect_times)
        redirects = request.meta.get('redirect_times', 0) + 1
        if ttl and redirects <= self.max_redirect_times:
            redirected.meta['redirect_times'] = redirects
            redirected.meta['redirect_ttl'] = ttl - 1
            redirected.meta['redirect_urls'] = request.meta.get('redirect_urls', []) + [request.url]
            redirected.meta['redirect_reasons'] = request.meta.get('redirect_reasons', []) + [reason]
            redirected.dont_filter = request.dont_filter
            redirected.priority = request.priority + self.priority_adjust
            logger.debug('Redirecting (%(reason)s) to %(redirected)s from %(request)s', {'reason': reason, 'redirected': redirected, 'request': request}, extra={'spider': spider})
            return redirected
        logger.debug('Discarding %(request)s: max redirections reached', {'request': request}, extra={'spider': spider})
        raise IgnoreRequest('max redirections reached')

    def _redirect_request_using_get(self, request, redirect_url):
        redirect_request = _build_redirect_request(request, url=redirect_url, method='GET', body='')
        redirect_request.headers.pop('Content-Type', None)
        redirect_request.headers.pop('Content-Length', None)
        return redirect_request