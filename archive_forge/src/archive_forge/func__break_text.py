import copy
import re
from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result
def _break_text(text, max_width, break_character):
    words = text.split()
    for word in words:
        if len(word) > max_width:
            replacement = _insert_break(word, max_width, break_character)
            text = text.replace(word, replacement)
    return text