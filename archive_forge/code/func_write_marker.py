import re
from formencode.rewritingparser import RewritingParser, html_quote
def write_marker(self, marker):
    self._content.append((marker,))