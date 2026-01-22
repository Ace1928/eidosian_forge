import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class WrapTransformation(object):
    """Wrap selection in an element."""

    def __init__(self, element):
        if isinstance(element, Element):
            self.element = element
        else:
            self.element = Element(element)

    def __call__(self, stream):
        for mark, event in stream:
            if mark:
                element = list(self.element.generate())
                for prefix in element[:-1]:
                    yield (None, prefix)
                yield (mark, event)
                start = mark
                stopped = False
                for mark, event in stream:
                    if start is ENTER and mark is EXIT:
                        yield (mark, event)
                        stopped = True
                        break
                    if not mark:
                        break
                    yield (mark, event)
                else:
                    stopped = True
                yield (None, element[-1])
                if not stopped:
                    yield (mark, event)
            else:
                yield (mark, event)