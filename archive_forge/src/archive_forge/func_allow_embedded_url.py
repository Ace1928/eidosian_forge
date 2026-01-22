import copy
import re
from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result
def allow_embedded_url(self, el, url):
    """
        Decide whether a URL that was found in an element's attributes or text
        if configured to be accepted or rejected.

        :param el: an element.
        :param url: a URL found on the element.
        :return: true to accept the URL and false to reject it.
        """
    if self.whitelist_tags is not None and el.tag not in self.whitelist_tags:
        return False
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https'):
        return False
    if parts.hostname in self.host_whitelist:
        return True
    return False