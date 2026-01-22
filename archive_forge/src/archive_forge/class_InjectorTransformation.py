import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class InjectorTransformation(object):
    """Abstract base class for transformations that inject content into a
    stream.

    >>> class Top(InjectorTransformation):
    ...     def __call__(self, stream):
    ...         for event in self._inject():
    ...             yield event
    ...         for event in stream:
    ...             yield event
    >>> html = HTML('<body>Some <em>test</em> text</body>', encoding='utf-8')
    >>> print(html | Transformer('.//em').apply(Top('Prefix ')))
    Prefix <body>Some <em>test</em> text</body>
    """

    def __init__(self, content):
        """Create a new injector.

        :param content: An iterable of Genshi stream events, or a string to be
                        injected.
        """
        self.content = content

    def _inject(self):
        content = self.content
        if hasattr(content, '__call__'):
            content = content()
        for event in _ensure(content):
            yield (None, event)