import re
import sys
from html import escape
from urllib.parse import quote
from paste.util.looper import looper
class HTMLTemplate(Template):
    default_namespace = Template.default_namespace.copy()
    default_namespace.update(dict(html=html, attr=attr, url=url))

    def _repr(self, value, pos):
        plain = Template._repr(self, value, pos)
        if isinstance(value, html):
            return plain
        else:
            return html_quote(plain)