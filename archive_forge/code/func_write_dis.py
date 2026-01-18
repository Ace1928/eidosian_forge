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
def write_dis(self, code_to_modify, op_number=None, prefix=''):
    filename, op_number = self._get_filename(op_number, prefix)
    with open(filename, 'w') as stream:
        stream.write('-------- ')
        stream.write('-------- ')
        stream.write('id(code_to_modify): %s' % id(code_to_modify))
        stream.write('\n\n')
        dis.dis(code_to_modify, file=stream)
    return op_number