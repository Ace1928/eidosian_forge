from collections import defaultdict
import itertools
import re
import warnings
import sys
from bs4.element import (
from . import _htmlparser
class HTMLTreeBuilder(TreeBuilder):
    """This TreeBuilder knows facts about HTML.

    Such as which tags are empty-element tags.
    """
    empty_element_tags = set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'keygen', 'link', 'menuitem', 'meta', 'param', 'source', 'track', 'wbr', 'basefont', 'bgsound', 'command', 'frame', 'image', 'isindex', 'nextid', 'spacer'])
    block_elements = set(['address', 'article', 'aside', 'blockquote', 'canvas', 'dd', 'div', 'dl', 'dt', 'fieldset', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 'hr', 'li', 'main', 'nav', 'noscript', 'ol', 'output', 'p', 'pre', 'section', 'table', 'tfoot', 'ul', 'video'])
    DEFAULT_STRING_CONTAINERS = {'rt': RubyTextString, 'rp': RubyParenthesisString, 'style': Stylesheet, 'script': Script, 'template': TemplateString}
    DEFAULT_CDATA_LIST_ATTRIBUTES = {'*': ['class', 'accesskey', 'dropzone'], 'a': ['rel', 'rev'], 'link': ['rel', 'rev'], 'td': ['headers'], 'th': ['headers'], 'td': ['headers'], 'form': ['accept-charset'], 'object': ['archive'], 'area': ['rel'], 'icon': ['sizes'], 'iframe': ['sandbox'], 'output': ['for']}
    DEFAULT_PRESERVE_WHITESPACE_TAGS = set(['pre', 'textarea'])

    def set_up_substitutions(self, tag):
        """Replace the declared encoding in a <meta> tag with a placeholder,
        to be substituted when the tag is output to a string.

        An HTML document may come in to Beautiful Soup as one
        encoding, but exit in a different encoding, and the <meta> tag
        needs to be changed to reflect this.

        :param tag: A `Tag`
        :return: Whether or not a substitution was performed.
        """
        if tag.name != 'meta':
            return False
        http_equiv = tag.get('http-equiv')
        content = tag.get('content')
        charset = tag.get('charset')
        meta_encoding = None
        if charset is not None:
            meta_encoding = charset
            tag['charset'] = CharsetMetaAttributeValue(charset)
        elif content is not None and http_equiv is not None and (http_equiv.lower() == 'content-type'):
            tag['content'] = ContentMetaAttributeValue(content)
        return meta_encoding is not None