import inspect
import linecache
import sys
import re
import os
from IPython import get_ipython
from contextlib import contextmanager
from IPython.utils import PyColorize
from IPython.utils import coloransi, py3compat
from IPython.core.excolors import exception_colors
from pdb import Pdb as OldPdb
def __format_line(self, tpl_line, filename, lineno, line, arrow=False):
    bp_mark = ''
    bp_mark_color = ''
    new_line, err = self.parser.format2(line, 'str')
    if not err:
        line = new_line
    bp = None
    if lineno in self.get_file_breaks(filename):
        bps = self.get_breaks(filename, lineno)
        bp = bps[-1]
    if bp:
        Colors = self.color_scheme_table.active_colors
        bp_mark = str(bp.number)
        bp_mark_color = Colors.breakpoint_enabled
        if not bp.enabled:
            bp_mark_color = Colors.breakpoint_disabled
    numbers_width = 7
    if arrow:
        pad = numbers_width - len(str(lineno)) - len(bp_mark)
        num = '%s%s' % (make_arrow(pad), str(lineno))
    else:
        num = '%*s' % (numbers_width - len(bp_mark), str(lineno))
    return tpl_line % (bp_mark_color + bp_mark, num, line)