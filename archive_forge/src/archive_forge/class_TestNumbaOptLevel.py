import os
import tempfile
from textwrap import dedent
import unittest
from unittest import mock
from numba.tests.support import (TestCase, temp_directory, override_env_config,
from numba.core import config
class TestNumbaOptLevel(TestCase):

    def check(self, expected, opt_value, raw_value):
        from numba import config, njit
        self.assertEqual(config.OPT, opt_value)
        self.assertEqual(config.OPT._raw_value, raw_value)
        from numba.core.codegen import CPUCodegen
        side_effect_message = 'expected side effect'

        def side_effect(*args, **kwargs):
            self.assertEqual(kwargs, expected)
            raise RuntimeError(side_effect_message)
        with mock.patch.object(CPUCodegen, '_module_pass_manager', side_effect=side_effect):
            with self.assertRaises(RuntimeError) as raises:
                njit(lambda: ...)()
            self.assertIn(side_effect_message, str(raises.exception))

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_OPT': 'max'})
    def test_opt_max(self):
        expected = {'loop_vectorize': True, 'slp_vectorize': False, 'opt': 3, 'cost': 'cheap'}
        self.check(expected, 3, 'max')

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_OPT': '3'})
    def test_opt_3(self):
        expected = {'loop_vectorize': False, 'slp_vectorize': False, 'opt': 0, 'cost': 'cheap'}
        self.check(expected, 3, 3)

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_OPT': '0'})
    def test_opt_0(self):
        expected = {'loop_vectorize': False, 'slp_vectorize': False, 'opt': 0, 'cost': 'cheap'}
        self.check(expected, 0, 0)

    @TestCase.run_test_in_subprocess()
    def test_opt_default(self):
        expected = {'loop_vectorize': False, 'slp_vectorize': False, 'opt': 0, 'cost': 'cheap'}
        self.check(expected, 3, 3)

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_OPT': 'invalid'})
    def test_opt_invalid(self):
        expected = {'loop_vectorize': False, 'slp_vectorize': False, 'opt': 0, 'cost': 'cheap'}
        self.check(expected, 3, 3)