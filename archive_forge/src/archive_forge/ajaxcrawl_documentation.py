import logging
import re
from w3lib import html
from scrapy.exceptions import NotConfigured
from scrapy.http import HtmlResponse

        Return True if a page without hash fragment could be "AJAX crawlable"
        according to https://developers.google.com/webmasters/ajax-crawling/docs/getting-started.
        