import functools
import os
import platform
import random
import string
import sys
import textwrap
import unittest
from os.path import join as pjoin
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
from IPython.core import debugger
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
import gc
def test_tclass(self):
    mydir = os.path.dirname(__file__)
    tc = os.path.join(mydir, 'tclass')
    src = f'import gc\n%run "{tc}" C-first\ngc.collect(0)\n%run "{tc}" C-second\ngc.collect(0)\n%run "{tc}" C-third\ngc.collect(0)\n%reset -f\n'
    self.mktmp(src, '.ipy')
    out = "ARGV 1-: ['C-first']\nARGV 1-: ['C-second']\ntclass.py: deleting object: C-first\nARGV 1-: ['C-third']\ntclass.py: deleting object: C-second\ntclass.py: deleting object: C-third\n"
    err = None
    tt.ipexec_validate(self.fname, out, err)