import html
from html.parser import HTMLParser
from django.utils.html import VOID_ELEMENTS
from django.utils.regex_helper import _lazy_re_compile
class RootElement(Element):

    def __init__(self):
        super().__init__(None, ())

    def __str__(self):
        return ''.join([html.escape(c) if isinstance(c, str) else str(c) for c in self.children])