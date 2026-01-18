import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def test_stderr_output(self) -> None:
    num_lines = 50
    with isolation.isolate(with_stderr_output) as env:
        pager = isolation.PagerControl(env)
        lines = num_lines
        while lines > 0:
            expected = min(lines, MAX_LINES_PER_PAGE)
            self.assertEqual(expected, pager.advance())
            lines -= expected
        self.assertEqual(0, pager.advance())
        self.assertEqual(0, pager.advance())
        self.assertEqual(0, pager.quit())
        self.assertEqual(num_lines, pager.total_lines())
        self.assertEqual('Hello world\n', env.error_output())
    self.assertEqual(0, env.exit_code())