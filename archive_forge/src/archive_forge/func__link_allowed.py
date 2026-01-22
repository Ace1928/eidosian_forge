import logging
import operator
from functools import partial
from urllib.parse import urljoin, urlparse
from lxml import etree
from parsel.csstranslator import HTMLTranslator
from w3lib.html import strip_html5_whitespace
from w3lib.url import canonicalize_url, safe_url_string
from scrapy.link import Link
from scrapy.linkextractors import (
from scrapy.utils.misc import arg_to_iter, rel_has_nofollow
from scrapy.utils.python import unique as unique_list
from scrapy.utils.response import get_base_url
from scrapy.utils.url import url_has_any_extension, url_is_from_any_domain
def _link_allowed(self, link):
    if not _is_valid_url(link.url):
        return False
    if self.allow_res and (not _matches(link.url, self.allow_res)):
        return False
    if self.deny_res and _matches(link.url, self.deny_res):
        return False
    parsed_url = urlparse(link.url)
    if self.allow_domains and (not url_is_from_any_domain(parsed_url, self.allow_domains)):
        return False
    if self.deny_domains and url_is_from_any_domain(parsed_url, self.deny_domains):
        return False
    if self.deny_extensions and url_has_any_extension(parsed_url, self.deny_extensions):
        return False
    if self.restrict_text and (not _matches(link.text, self.restrict_text)):
        return False
    return True