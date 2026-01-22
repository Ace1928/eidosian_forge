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
class ListTB(TBTools):
    """Print traceback information from a traceback list, with optional color.

    Calling requires 3 arguments: (etype, evalue, elist)
    as would be obtained by::

      etype, evalue, tb = sys.exc_info()
      if tb:
        elist = traceback.extract_tb(tb)
      else:
        elist = None

    It can thus be used by programs which need to process the traceback before
    printing (such as console replacements based on the code module from the
    standard library).

    Because they are meant to be called without a full traceback (only a
    list), instances of this class can't call the interactive pdb debugger."""

    def __call__(self, etype, value, elist):
        self.ostream.flush()
        self.ostream.write(self.text(etype, value, elist))
        self.ostream.write('\n')

    def _extract_tb(self, tb):
        if tb:
            return traceback.extract_tb(tb)
        else:
            return None

    def structured_traceback(self, etype: type, evalue: Optional[BaseException], etb: Optional[TracebackType]=None, tb_offset: Optional[int]=None, context=5):
        """Return a color formatted string with the traceback info.

        Parameters
        ----------
        etype : exception type
            Type of the exception raised.
        evalue : object
            Data stored in the exception
        etb : list | TracebackType | None
            If list: List of frames, see class docstring for details.
            If Traceback: Traceback of the exception.
        tb_offset : int, optional
            Number of frames in the traceback to skip.  If not given, the
            instance evalue is used (set in constructor).
        context : int, optional
            Number of lines of context information to print.

        Returns
        -------
        String with formatted exception.
        """
        if isinstance(etb, tuple):
            etb, chained_exc_ids = etb
        else:
            chained_exc_ids = set()
        if isinstance(etb, list):
            elist = etb
        elif etb is not None:
            elist = self._extract_tb(etb)
        else:
            elist = []
        tb_offset = self.tb_offset if tb_offset is None else tb_offset
        assert isinstance(tb_offset, int)
        Colors = self.Colors
        out_list = []
        if elist:
            if tb_offset and len(elist) > tb_offset:
                elist = elist[tb_offset:]
            out_list.append('Traceback %s(most recent call last)%s:' % (Colors.normalEm, Colors.Normal) + '\n')
            out_list.extend(self._format_list(elist))
        lines = ''.join(self._format_exception_only(etype, evalue))
        out_list.append(lines)
        exception = self.get_parts_of_chained_exception(evalue)
        if exception and id(exception[1]) not in chained_exc_ids:
            chained_exception_message = self.prepare_chained_exception_message(evalue.__cause__)[0] if evalue is not None else ''
            etype, evalue, etb = exception
            chained_exc_ids.add(id(exception[1]))
            chained_exceptions_tb_offset = 0
            out_list = self.structured_traceback(etype, evalue, (etb, chained_exc_ids), chained_exceptions_tb_offset, context) + chained_exception_message + out_list
        return out_list

    def _format_list(self, extracted_list):
        """Format a list of traceback entry tuples for printing.

        Given a list of tuples as returned by extract_tb() or
        extract_stack(), return a list of strings ready for printing.
        Each string in the resulting list corresponds to the item with the
        same index in the argument list.  Each string ends in a newline;
        the strings may contain internal newlines as well, for those items
        whose source text line is not None.

        Lifted almost verbatim from traceback.py
        """
        Colors = self.Colors
        output_list = []
        for ind, (filename, lineno, name, line) in enumerate(extracted_list):
            normalCol, nameCol, fileCol, lineCol = (Colors.normalEm, Colors.nameEm, Colors.filenameEm, Colors.line) if ind == len(extracted_list) - 1 else (Colors.Normal, Colors.name, Colors.filename, '')
            fns = _format_filename(filename, fileCol, normalCol, lineno=lineno)
            item = f'{normalCol}  {fns}'
            if name != '<module>':
                item += f' in {nameCol}{name}{normalCol}\n'
            else:
                item += '\n'
            if line:
                item += f'{lineCol}    {line.strip()}{normalCol}\n'
            output_list.append(item)
        return output_list

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

    def get_exception_only(self, etype, value):
        """Only print the exception type and message, without a traceback.

        Parameters
        ----------
        etype : exception type
        value : exception value
        """
        return ListTB.structured_traceback(self, etype, value)

    def show_exception_only(self, etype, evalue):
        """Only print the exception type and message, without a traceback.

        Parameters
        ----------
        etype : exception type
        evalue : exception value
        """
        ostream = self.ostream
        ostream.flush()
        ostream.write('\n'.join(self.get_exception_only(etype, evalue)))
        ostream.flush()

    def _some_str(self, value):
        try:
            return py3compat.cast_unicode(str(value))
        except:
            return u'<unprintable %s object>' % type(value).__name__