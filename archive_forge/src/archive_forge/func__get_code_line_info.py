from collections import namedtuple
import dis
from functools import partial
import itertools
import os.path
import sys
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import Instr, Label
from _pydev_bundle import pydev_log
from _pydevd_frame_eval.pydevd_frame_tracing import _pydev_stop_at_break, _pydev_needs_stop_at_break
def _get_code_line_info(code_obj):
    line_to_offset = {}
    first_line = None
    last_line = None
    for offset, line in dis.findlinestarts(code_obj):
        line_to_offset[line] = offset
    if line_to_offset:
        first_line = min(line_to_offset)
        last_line = max(line_to_offset)
    return _CodeLineInfo(line_to_offset, first_line, last_line)