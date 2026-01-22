import logging
import re
from w3lib import html
from scrapy.exceptions import NotConfigured
from scrapy.http import HtmlResponse
class AjaxCrawlMiddleware:
    """
    Handle 'AJAX crawlable' pages marked as crawlable via meta tag.
    For more info see https://developers.google.com/webmasters/ajax-crawling/docs/getting-started.
    """

    def __init__(self, settings):
        if not settings.getbool('AJAXCRAWL_ENABLED'):
            raise NotConfigured
        self.lookup_bytes = settings.getint('AJAXCRAWL_MAXSIZE', 32768)

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)

    def process_response(self, request, response, spider):
        if not isinstance(response, HtmlResponse) or response.status != 200:
            return response
        if request.method != 'GET':
            return response
        if 'ajax_crawlable' in request.meta:
            return response
        if not self._has_ajax_crawlable_variant(response):
            return response
        ajax_crawl_request = request.replace(url=request.url + '#!')
        logger.debug('Downloading AJAX crawlable %(ajax_crawl_request)s instead of %(request)s', {'ajax_crawl_request': ajax_crawl_request, 'request': request}, extra={'spider': spider})
        ajax_crawl_request.meta['ajax_crawlable'] = True
        return ajax_crawl_request

    def _has_ajax_crawlable_variant(self, response):
        """
        Return True if a page without hash fragment could be "AJAX crawlable"
        according to https://developers.google.com/webmasters/ajax-crawling/docs/getting-started.
        """
        body = response.text[:self.lookup_bytes]
        return _has_ajaxcrawlable_meta(body)