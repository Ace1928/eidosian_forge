import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class SubstituteTransformation(object):
    """Replace text matching a regular expression.

    Refer to the documentation for ``re.sub()`` for details.
    """

    def __init__(self, pattern, replace, count=0):
        """Create the transform.

        :param pattern: A regular expression object, or string.
        :param replace: Replacement pattern.
        :param count: Number of replacements to make in each text fragment.
        """
        if isinstance(pattern, six.string_types):
            self.pattern = re.compile(pattern)
        else:
            self.pattern = pattern
        self.count = count
        self.replace = replace

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: The marked event stream to filter
        """
        for mark, (kind, data, pos) in stream:
            if mark is not None and kind is TEXT:
                new_data = self.pattern.sub(self.replace, data, self.count)
                if isinstance(data, Markup):
                    data = Markup(new_data)
                else:
                    data = new_data
            yield (mark, (kind, data, pos))