import os
import tempfile
from textwrap import dedent
import unittest
from unittest import mock
from numba.tests.support import (TestCase, temp_directory, override_env_config,
from numba.core import config
@TestCase.run_test_in_subprocess(envvars={'NUMBA_OPT': 'max'})
def test_opt_max(self):
    expected = {'loop_vectorize': True, 'slp_vectorize': False, 'opt': 3, 'cost': 'cheap'}
    self.check(expected, 3, 'max')