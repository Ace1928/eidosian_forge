from io import BytesIO
from typing import Callable, Dict, List, Tuple
from .. import errors, osutils, registry
class ContentFilter:

    def __init__(self, reader, writer):
        """Create a filter that converts content while reading and writing.

        Args:
          reader: function for converting convenience to canonical content
          writer: function for converting canonical to convenience content
        """
        self.reader = reader
        self.writer = writer

    def __repr__(self):
        return 'reader: {}, writer: {}'.format(self.reader, self.writer)