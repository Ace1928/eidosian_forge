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
def with_fake_debugger(func):

    @functools.wraps(func)
    def wrapper(*args, **kwds):
        with patch.object(debugger.Pdb, 'run', staticmethod(eval)):
            return func(*args, **kwds)
    return wrapper