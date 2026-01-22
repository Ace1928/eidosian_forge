import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class ReplaceTransformation(InjectorTransformation):
    """Replace selection with content."""

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: The marked event stream to filter
        """
        stream = PushBackStream(stream)
        for mark, event in stream:
            if mark is not None:
                start = mark
                for subevent in self._inject():
                    yield subevent
                for mark, event in stream:
                    if start is ENTER:
                        if mark is EXIT:
                            break
                    elif mark != start:
                        stream.push((mark, event))
                        break
            else:
                yield (mark, event)