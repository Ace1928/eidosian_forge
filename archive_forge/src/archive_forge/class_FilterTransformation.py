import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class FilterTransformation(object):
    """Apply a normal stream filter to the selection. The filter is called once
    for each selection."""

    def __init__(self, filter):
        """Create the transform.

        :param filter: The stream filter to apply.
        """
        self.filter = filter

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: The marked event stream to filter
        """

        def flush(queue):
            if queue:
                for event in self.filter(queue):
                    yield (OUTSIDE, event)
                del queue[:]
        queue = []
        for mark, event in stream:
            if mark is ENTER:
                queue.append(event)
                for mark, event in stream:
                    queue.append(event)
                    if mark is EXIT:
                        break
                for queue_event in flush(queue):
                    yield queue_event
            elif mark is OUTSIDE:
                stopped = False
                queue.append(event)
                for mark, event in stream:
                    if mark is not OUTSIDE:
                        break
                    queue.append(event)
                else:
                    stopped = True
                for queue_event in flush(queue):
                    yield queue_event
                if not stopped:
                    yield (mark, event)
            else:
                yield (mark, event)
        for queue_event in flush(queue):
            yield queue_event