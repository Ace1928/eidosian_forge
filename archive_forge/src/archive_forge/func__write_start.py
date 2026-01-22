from html.parser import HTMLParser
from itertools import zip_longest
def _write_start(self, doc):
    handler_name = 'start_%s' % self.tag
    if hasattr(doc.style, handler_name):
        getattr(doc.style, handler_name)(self.attrs)