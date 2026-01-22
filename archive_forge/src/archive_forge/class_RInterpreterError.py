import contextlib
import sys
import tempfile
from glob import glob
import os
from shutil import rmtree
import textwrap
import typing
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface as ri
import rpy2.rinterface_lib.openrlib
import rpy2.robjects as ro
import rpy2.robjects.packages as rpacks
from rpy2.robjects.lib import grdevices
from rpy2.robjects.conversion import (Converter,
import warnings
import IPython.display  # type: ignore
from IPython.core import displaypub  # type: ignore
from IPython.core.magic import (Magics,   # type: ignore
from IPython.core.magic_arguments import (argument,  # type: ignore
class RInterpreterError(ri.embedded.RRuntimeError):
    """An error when running R code in a %%R magic cell."""
    msg_prefix_template = 'Failed to parse and evaluate line %r.\nR error message: %r'
    rstdout_prefix = '\nR stdout:\n'

    def __init__(self, line, err, stdout):
        self.line = line
        self.err = err.rstrip()
        self.stdout = stdout.rstrip()

    def __str__(self):
        s = self.msg_prefix_template % (self.line, self.err)
        if self.stdout and self.stdout != self.err:
            s += self.rstdout_prefix + self.stdout
        return s