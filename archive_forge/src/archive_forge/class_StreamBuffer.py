import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class StreamBuffer(Stream):
    """Stream event buffer used for cut and copy transformations."""

    def __init__(self):
        """Create the buffer."""
        Stream.__init__(self, [])

    def append(self, event):
        """Add an event to the buffer.

        :param event: the markup event to add
        """
        self.events.append(event)

    def reset(self):
        """Empty the buffer of events."""
        del self.events[:]