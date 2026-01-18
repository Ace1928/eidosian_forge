import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def test_page_to_end(self) -> None:
    num_lines = 100
    with isolation.isolate(finite(num_lines)) as env:
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
        self.assertFalse(env.error_output())
    self.assertEqual(0, env.exit_code())