from __future__ import unicode_literals
import re
from xml.sax.saxutils import unescape
from tensorboard._vendor import html5lib
from tensorboard._vendor.html5lib.constants import namespaces
from tensorboard._vendor.html5lib.filters import sanitizer
from tensorboard._vendor.html5lib.serializer import HTMLSerializer
from tensorboard._vendor.bleach.encoding import force_unicode
from tensorboard._vendor.bleach.utils import alphabetize_attributes
def allow_token(self, token):
    """Handles the case where we're allowing the tag"""
    if 'data' in token:
        attrs = {}
        for namespaced_name, val in token['data'].items():
            namespace, name = namespaced_name
            if not self.attr_filter(token['name'], name, val):
                continue
            if namespaced_name in self.attr_val_is_uri:
                val_unescaped = re.sub('[`\x00- \x7f-\xa0\\s]+', '', unescape(val)).lower()
                val_unescaped = val_unescaped.replace('�', '')
                if re.match('^[a-z0-9][-+.a-z0-9]*:', val_unescaped) and val_unescaped.split(':')[0] not in self.allowed_protocols:
                    continue
            if namespaced_name in self.svg_attr_val_allows_ref:
                new_val = re.sub('url\\s*\\(\\s*[^#\\s][^)]+?\\)', ' ', unescape(val))
                new_val = new_val.strip()
                if not new_val:
                    continue
                else:
                    val = new_val
            if (None, token['name']) in self.svg_allow_local_href:
                if namespaced_name in [(None, 'href'), (namespaces['xlink'], 'href')]:
                    if re.search('^\\s*[^#\\s]', val):
                        continue
            if namespaced_name == (None, u'style'):
                val = self.sanitize_css(val)
            attrs[namespaced_name] = val
        token['data'] = alphabetize_attributes(attrs)
    return token