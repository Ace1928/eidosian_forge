import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class InvertTransformation(object):
    """Invert selection so that marked events become unmarked, and vice versa.

    Specificaly, all input marks are converted to null marks, and all input
    null marks are converted to OUTSIDE marks.
    """

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: the marked event stream to filter
        """
        for mark, event in stream:
            if mark:
                yield (None, event)
            else:
                yield (OUTSIDE, event)