from collections.abc import Sequence
import functools
import inspect
import linecache
import pydoc
import sys
import time
import traceback
import types
from types import TracebackType
from typing import Any, List, Optional, Tuple
import stack_data
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.styles import get_style_by_name
import IPython.utils.colorable as colorable
from IPython import get_ipython
from IPython.core import debugger
from IPython.core.display_trap import DisplayTrap
from IPython.core.excolors import exception_colors
from IPython.utils import PyColorize
from IPython.utils import path as util_path
from IPython.utils import py3compat
from IPython.utils.terminal import get_terminal_size
def _format_exception_only(self, etype, value):
    """Format the exception part of a traceback.

        The arguments are the exception type and value such as given by
        sys.exc_info()[:2]. The return value is a list of strings, each ending
        in a newline.  Normally, the list contains a single string; however,
        for SyntaxError exceptions, it contains several lines that (when
        printed) display detailed information about where the syntax error
        occurred.  The message indicating which exception occurred is the
        always last string in the list.

        Also lifted nearly verbatim from traceback.py
        """
    have_filedata = False
    Colors = self.Colors
    output_list = []
    stype = py3compat.cast_unicode(Colors.excName + etype.__name__ + Colors.Normal)
    if value is None:
        output_list.append(stype + '\n')
    else:
        if issubclass(etype, SyntaxError):
            have_filedata = True
            if not value.filename:
                value.filename = '<string>'
            if value.lineno:
                lineno = value.lineno
                textline = linecache.getline(value.filename, value.lineno)
            else:
                lineno = 'unknown'
                textline = ''
            output_list.append('%s  %s%s\n' % (Colors.normalEm, _format_filename(value.filename, Colors.filenameEm, Colors.normalEm, lineno=None if lineno == 'unknown' else lineno), Colors.Normal))
            if textline == '':
                textline = py3compat.cast_unicode(value.text, 'utf-8')
            if textline is not None:
                i = 0
                while i < len(textline) and textline[i].isspace():
                    i += 1
                output_list.append('%s    %s%s\n' % (Colors.line, textline.strip(), Colors.Normal))
                if value.offset is not None:
                    s = '    '
                    for c in textline[i:value.offset - 1]:
                        if c.isspace():
                            s += c
                        else:
                            s += ' '
                    output_list.append('%s%s^%s\n' % (Colors.caret, s, Colors.Normal))
        try:
            s = value.msg
        except Exception:
            s = self._some_str(value)
        if s:
            output_list.append('%s%s:%s %s\n' % (stype, Colors.excName, Colors.Normal, s))
        else:
            output_list.append('%s\n' % stype)
        output_list.extend((f'{x}\n' for x in getattr(value, '__notes__', [])))
    if have_filedata:
        ipinst = get_ipython()
        if ipinst is not None:
            ipinst.hooks.synchronize_with_editor(value.filename, value.lineno, 0)
    return output_list