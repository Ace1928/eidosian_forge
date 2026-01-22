from __future__ import annotations
import os
import traceback
from typing import TYPE_CHECKING, Final, cast
import streamlit
from streamlit.errors import (
from streamlit.logger import get_logger
from streamlit.proto.Exception_pb2 import Exception as ExceptionProto
from streamlit.runtime.metrics_util import gather_metrics
def _format_syntax_error_message(exception: SyntaxError) -> str:
    """Returns a nicely formatted SyntaxError message that emulates
    what the Python interpreter outputs, e.g.:

    > File "raven.py", line 3
    >   st.write('Hello world!!'))
    >                            ^
    > SyntaxError: invalid syntax

    """
    if exception.text:
        if exception.offset is not None:
            caret_indent = ' ' * max(exception.offset - 1, 0)
        else:
            caret_indent = ''
        return 'File "%(filename)s", line %(lineno)s\n  %(text)s\n  %(caret_indent)s^\n%(errname)s: %(msg)s' % {'filename': exception.filename, 'lineno': exception.lineno, 'text': exception.text.rstrip(), 'caret_indent': caret_indent, 'errname': type(exception).__name__, 'msg': exception.msg}
    return str(exception)