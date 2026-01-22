import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class MapTransformation(object):
    """Apply a function to the `data` element of events of ``kind`` in the
    selection.
    """

    def __init__(self, function, kind):
        """Create the transform.

        :param function: the function to apply; the function must take one
                         argument, the `data` element of each selected event
        :param kind: the stream event ``kind`` to apply the `function` to
        """
        self.function = function
        self.kind = kind

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: The marked event stream to filter
        """
        for mark, (kind, data, pos) in stream:
            if mark and self.kind in (None, kind):
                yield (mark, (kind, self.function(data), pos))
            else:
                yield (mark, (kind, data, pos))