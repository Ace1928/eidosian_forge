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
def _map_to_standard_prefix(self, name):
    colonpos = name.find(':')
    if colonpos != -1:
        prefix = name[:colonpos]
        suffix = name[colonpos + 1:]
        prefix = self.namespacemap.get(prefix, prefix)
        name = prefix + ':' + suffix
    return name