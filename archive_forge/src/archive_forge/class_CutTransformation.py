import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class CutTransformation(object):
    """Cut selected events into a buffer for later insertion and remove the
    selection.
    """

    def __init__(self, buffer, accumulate=False):
        """Create the cut transformation.

        :param buffer: the `StreamBuffer` in which the selection should be
                       stored
        """
        self.buffer = buffer
        self.accumulate = accumulate

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: the marked event stream to filter
        """
        attributes = []
        stream = PushBackStream(stream)
        broken = False
        if not self.accumulate:
            self.buffer.reset()
        for mark, event in stream:
            if mark:
                if not self.accumulate:
                    if not broken and self.buffer:
                        yield (BREAK, (BREAK, None, None))
                    self.buffer.reset()
                self.buffer.append(event)
                start = mark
                if mark is ATTR:
                    attributes.extend([name for name, _ in event[1][1]])
                for mark, event in stream:
                    if start is mark is ATTR:
                        attributes.extend([name for name, _ in event[1][1]])
                    if start is not ENTER and mark != start:
                        if start is ATTR:
                            kind, data, pos = event
                            assert kind is START
                            data = (data[0], data[1] - attributes)
                            attributes = None
                            stream.push((mark, (kind, data, pos)))
                        else:
                            stream.push((mark, event))
                        break
                    self.buffer.append(event)
                    if start is ENTER and mark is EXIT:
                        break
                broken = False
            else:
                broken = True
                yield (mark, event)
        if not broken and self.buffer:
            yield (BREAK, (BREAK, None, None))