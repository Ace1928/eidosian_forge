from html.parser import HTMLParser
from itertools import zip_longest
def _doc_has_handler(self, tag, is_start):
    if is_start:
        handler_name = 'start_%s' % tag
    else:
        handler_name = 'end_%s' % tag
    return hasattr(self.doc.style, handler_name)