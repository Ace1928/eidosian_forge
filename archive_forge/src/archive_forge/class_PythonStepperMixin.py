from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class PythonStepperMixin(object):
    """
    Make this a mixin so CyStep can also inherit from this and use a
    CythonCodeStepper at the same time.
    """

    def python_step(self, stepinto):
        """
        Set a watchpoint on the Python bytecode instruction pointer and try
        to finish the frame
        """
        output = gdb.execute('watch f->f_lasti', to_string=True)
        watchpoint = int(re.search('[Ww]atchpoint (\\d+):', output).group(1))
        self.step(stepinto=stepinto, stepover_command='finish')
        gdb.execute('delete %s' % watchpoint)