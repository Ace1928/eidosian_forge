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
def unknown_endtag(self, tag):
    if tag.find(':') != -1:
        prefix, suffix = tag.split(':', 1)
    else:
        prefix, suffix = ('', tag)
    prefix = self.namespacemap.get(prefix, prefix)
    if prefix:
        prefix = prefix + '_'
    if suffix == 'svg' and self.svgOK:
        self.svgOK -= 1
    methodname = '_end_' + prefix + suffix
    try:
        if self.svgOK:
            raise AttributeError()
        method = getattr(self, methodname)
        method()
    except AttributeError:
        self.pop(prefix + suffix)
    if self.incontent and (not self.contentparams.get('type', 'xml').endswith('xml')):
        if tag in ('xhtml:div', 'div'):
            return
        self.contentparams['type'] = 'application/xhtml+xml'
    if self.incontent and self.contentparams.get('type') == 'application/xhtml+xml':
        tag = tag.split(':')[-1]
        self.handle_data('</%s>' % tag, escape=0)
    if self.basestack:
        self.basestack.pop()
        if self.basestack and self.basestack[-1]:
            self.baseuri = self.basestack[-1]
    if self.langstack:
        self.langstack.pop()
        if self.langstack:
            self.lang = self.langstack[-1]
    self.depth -= 1