import re
from email import errors
from email._policybase import compat32
from collections import deque
from io import StringIO
class BytesFeedParser(FeedParser):
    """Like FeedParser, but feed accepts bytes."""

    def feed(self, data):
        super().feed(data.decode('ascii', 'surrogateescape'))