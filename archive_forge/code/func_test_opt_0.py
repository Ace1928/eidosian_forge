import os
import tempfile
from textwrap import dedent
import unittest
from unittest import mock
from numba.tests.support import (TestCase, temp_directory, override_env_config,
from numba.core import config
@TestCase.run_test_in_subprocess(envvars={'NUMBA_OPT': '0'})
def test_opt_0(self):
    expected = {'loop_vectorize': False, 'slp_vectorize': False, 'opt': 0, 'cost': 'cheap'}
    self.check(expected, 0, 0)