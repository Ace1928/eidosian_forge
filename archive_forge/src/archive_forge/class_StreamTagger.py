import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
class StreamTagger(CopyStreamResult):
    """Adds or discards tags from StreamResult events."""

    def __init__(self, targets, add=None, discard=None):
        """Create a StreamTagger.

        :param targets: A list of targets to forward events onto.
        :param add: Either None or an iterable of tags to add to each event.
        :param discard: Either None or an iterable of tags to discard from each
            event.
        """
        super().__init__(targets)
        self.add = frozenset(add or ())
        self.discard = frozenset(discard or ())

    def status(self, *args, **kwargs):
        test_tags = kwargs.get('test_tags') or set()
        test_tags.update(self.add)
        test_tags.difference_update(self.discard)
        kwargs['test_tags'] = test_tags or None
        super().status(*args, **kwargs)