import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def test_short_output_reset(self) -> None:
    num_lines = 10
    with isolation.isolate(finite(num_lines, reset_on_exit=True)) as env:
        pager = isolation.PagerControl(env)
        self.assertEqual(num_lines, pager.quit())
        self.assertFalse(env.error_output())
    self.assertEqual(0, env.exit_code())