from __future__ import unicode_literals
import re
from xml.sax.saxutils import unescape
from tensorboard._vendor import html5lib
from tensorboard._vendor.html5lib.constants import namespaces
from tensorboard._vendor.html5lib.filters import sanitizer
from tensorboard._vendor.html5lib.serializer import HTMLSerializer
from tensorboard._vendor.bleach.encoding import force_unicode
from tensorboard._vendor.bleach.utils import alphabetize_attributes
class BleachSanitizerFilter(sanitizer.Filter):
    """html5lib Filter that sanitizes text

    This filter can be used anywhere html5lib filters can be used.

    """

    def __init__(self, source, attributes=ALLOWED_ATTRIBUTES, strip_disallowed_elements=False, strip_html_comments=True, **kwargs):
        """Creates a BleachSanitizerFilter instance

        :arg Treewalker source: stream

        :arg list tags: allowed list of tags; defaults to
            ``bleach.sanitizer.ALLOWED_TAGS``

        :arg dict attributes: allowed attributes; can be a callable, list or dict;
            defaults to ``bleach.sanitizer.ALLOWED_ATTRIBUTES``

        :arg list styles: allowed list of css styles; defaults to
            ``bleach.sanitizer.ALLOWED_STYLES``

        :arg list protocols: allowed list of protocols for links; defaults
            to ``bleach.sanitizer.ALLOWED_PROTOCOLS``

        :arg bool strip_disallowed_elements: whether or not to strip disallowed
            elements

        :arg bool strip_html_comments: whether or not to strip HTML comments

        """
        self.attr_filter = attribute_filter_factory(attributes)
        self.strip_disallowed_elements = strip_disallowed_elements
        self.strip_html_comments = strip_html_comments
        return super(BleachSanitizerFilter, self).__init__(source, **kwargs)

    def sanitize_token(self, token):
        """Sanitize a token either by HTML-encoding or dropping.

        Unlike sanitizer.Filter, allowed_attributes can be a dict of {'tag':
        ['attribute', 'pairs'], 'tag': callable}.

        Here callable is a function with two arguments of attribute name and
        value. It should return true of false.

        Also gives the option to strip tags instead of encoding.

        """
        token_type = token['type']
        if token_type in ['StartTag', 'EndTag', 'EmptyTag']:
            if token['name'] in self.allowed_elements:
                return self.allow_token(token)
            elif self.strip_disallowed_elements:
                pass
            else:
                if 'data' in token:
                    token['data'] = alphabetize_attributes(token['data'])
                return self.disallowed_token(token)
        elif token_type == 'Comment':
            if not self.strip_html_comments:
                return token
        else:
            return token

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

    def sanitize_css(self, style):
        """Sanitizes css in style tags"""
        style = re.compile('url\\s*\\(\\s*[^\\s)]+?\\s*\\)\\s*').sub(' ', style)
        parts = style.split(';')
        gauntlet = re.compile('^([-/:,#%.\'"\\sa-zA-Z0-9!]|\\w-\\w|\'[\\s\\w]+\'\\s*|"[\\s\\w]+"|\\([\\d,%\\.\\s]+\\))*$')
        for part in parts:
            if not gauntlet.match(part):
                return ''
        if not re.match('^\\s*([-\\w]+\\s*:[^:;]*(;\\s*|$))*$', style):
            return ''
        clean = []
        for prop, value in re.findall('([-\\w]+)\\s*:\\s*([^:;]*)', style):
            if not value:
                continue
            if prop.lower() in self.allowed_css_properties:
                clean.append(prop + ': ' + value + ';')
            elif prop.lower() in self.allowed_svg_properties:
                clean.append(prop + ': ' + value + ';')
        return ' '.join(clean)