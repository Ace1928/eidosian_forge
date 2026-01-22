import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
def after(self, content):
    """Insert content after selection.

        Here, we insert some text after the </em> closing tag:

        >>> html = HTML('<html><head><title>Some Title</title></head>'
        ...             '<body>Some <em>body</em> text.</body></html>',
        ...             encoding='utf-8')
        >>> print(html | Transformer('.//em').after(' rock'))
        <html><head><title>Some Title</title></head><body>Some <em>body</em>
        rock text.</body></html>

        :param content: Either a callable, an iterable of events, or a string
                        to insert.
        :rtype: `Transformer`
        """
    return self.apply(AfterTransformation(content))