import codecs
import functools
import json
import os
import traceback
from testtools.compat import _b
from testtools.content_type import ContentType, JSON, UTF8_TEXT
def StacktraceContent(prefix_content='', postfix_content=''):
    """Content object for stack traces.

    This function will create and return a 'Content' object that contains a
    stack trace.

    The mime type is set to 'text/x-traceback;language=python', so other
    languages can format their stack traces differently.

    :param prefix_content: A unicode string to add before the stack lines.
    :param postfix_content: A unicode string to add after the stack lines.
    """
    stack = traceback.walk_stack(None)

    def filter_stack(stack):
        next(stack)
        next(stack)
        for f, f_lineno in stack:
            if StackLinesContent.HIDE_INTERNAL_STACK:
                if '__unittest' in f.f_globals:
                    return
                yield (f, f_lineno)
    extract = traceback.StackSummary.extract(filter_stack(stack))
    extract.reverse()
    return StackLinesContent(extract, prefix_content, postfix_content)