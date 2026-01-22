import codecs
import functools
import json
import os
import traceback
from testtools.compat import _b
from testtools.content_type import ContentType, JSON, UTF8_TEXT
def _stack_lines_to_unicode(self, stack_lines):
    """Converts a list of pre-processed stack lines into a unicode string.
        """
    msg_lines = traceback.format_list(stack_lines)
    return ''.join(msg_lines)