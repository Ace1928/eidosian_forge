import logging
import re
from w3lib import html
from scrapy.exceptions import NotConfigured
from scrapy.http import HtmlResponse
def _has_ajax_crawlable_variant(self, response):
    """
        Return True if a page without hash fragment could be "AJAX crawlable"
        according to https://developers.google.com/webmasters/ajax-crawling/docs/getting-started.
        """
    body = response.text[:self.lookup_bytes]
    return _has_ajaxcrawlable_meta(body)