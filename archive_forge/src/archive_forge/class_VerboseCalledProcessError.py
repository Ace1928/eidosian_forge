from __future__ import print_function, absolute_import
import os
import tempfile
import unittest
import sys
import re
import warnings
import io
from textwrap import dedent
from future.utils import bind_method, PY26, PY3, PY2, PY27
from future.moves.subprocess import check_output, STDOUT, CalledProcessError
class VerboseCalledProcessError(CalledProcessError):
    """
    Like CalledProcessError, but it displays more information (message and
    script output) for diagnosing test failures etc.
    """

    def __init__(self, msg, returncode, cmd, output=None):
        self.msg = msg
        self.returncode = returncode
        self.cmd = cmd
        self.output = output

    def __str__(self):
        return "Command '%s' failed with exit status %d\nMessage: %s\nOutput: %s" % (self.cmd, self.returncode, self.msg, self.output)