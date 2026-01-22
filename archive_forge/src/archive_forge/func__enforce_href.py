import base64
import binascii
import copy
import html.entities
import re
import xml.sax.saxutils
from .html import _cp1252
from .namespaces import _base, cc, dc, georss, itunes, mediarss, psc
from .sanitizer import _sanitize_html, _HTMLSanitizer
from .util import FeedParserDict
from .urls import _urljoin, make_safe_absolute_uri, resolve_relative_uris
@staticmethod
def _enforce_href(attrs_d):
    href = attrs_d.get('url', attrs_d.get('uri', attrs_d.get('href', None)))
    if href:
        try:
            del attrs_d['url']
        except KeyError:
            pass
        try:
            del attrs_d['uri']
        except KeyError:
            pass
        attrs_d['href'] = href
    return attrs_d