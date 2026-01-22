import codecs
import functools
import json
import os
import traceback
from testtools.compat import _b
from testtools.content_type import ContentType, JSON, UTF8_TEXT
class StackLinesContent(Content):
    """Content object for stack lines.

    This adapts a list of "preprocessed" stack lines into a 'Content' object.
    The stack lines are most likely produced from ``traceback.extract_stack``
    or ``traceback.extract_tb``.

    text/x-traceback;language=python is used for the mime type, in order to
    provide room for other languages to format their tracebacks differently.
    """
    HIDE_INTERNAL_STACK = True

    def __init__(self, stack_lines, prefix_content='', postfix_content=''):
        """Create a StackLinesContent for ``stack_lines``.

        :param stack_lines: A list of preprocessed stack lines, probably
            obtained by calling ``traceback.extract_stack`` or
            ``traceback.extract_tb``.
        :param prefix_content: If specified, a unicode string to prepend to the
            text content.
        :param postfix_content: If specified, a unicode string to append to the
            text content.
        """
        content_type = ContentType('text', 'x-traceback', {'language': 'python', 'charset': 'utf8'})
        value = prefix_content + self._stack_lines_to_unicode(stack_lines) + postfix_content
        super().__init__(content_type, lambda: [value.encode('utf8')])

    def _stack_lines_to_unicode(self, stack_lines):
        """Converts a list of pre-processed stack lines into a unicode string.
        """
        msg_lines = traceback.format_list(stack_lines)
        return ''.join(msg_lines)