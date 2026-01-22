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
class PyStep(ExecutionControlCommandBase, PythonStepperMixin):
    """Step through Python code."""
    stepinto = True

    @dont_suppress_errors
    def invoke(self, args, from_tty):
        self.python_step(stepinto=self.stepinto)