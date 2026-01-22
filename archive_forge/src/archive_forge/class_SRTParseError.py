from __future__ import unicode_literals
import functools
import re
from datetime import timedelta
import logging
import io
class SRTParseError(Exception):
    """
    Raised when part of an SRT block could not be parsed.

    :param int expected_start: The expected contiguous start index
    :param int actual_start: The actual non-contiguous start index
    :param str unmatched_content: The content between the expected start index
                                  and the actual start index
    """

    def __init__(self, expected_start, actual_start, unmatched_content):
        message = 'Expected contiguous start of match or end of input at char %d, but started at char %d (unmatched content: %r)' % (expected_start, actual_start, unmatched_content)
        super(SRTParseError, self).__init__(message)
        self.expected_start = expected_start
        self.actual_start = actual_start
        self.unmatched_content = unmatched_content