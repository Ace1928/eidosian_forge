import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
class InvokePagerTest(unittest.TestCase):

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

    def test_page_to_middle(self) -> None:
        num_lines = 100
        with isolation.isolate(finite(num_lines)) as env:
            pager = isolation.PagerControl(env)
            self.assertEqual(MAX_LINES_PER_PAGE, pager.advance())
            self.assertEqual(MAX_LINES_PER_PAGE, pager.advance())
            self.assertEqual(MAX_LINES_PER_PAGE, pager.quit())
            self.assertFalse(env.error_output())
        self.assertEqual(0, env.exit_code())

    def test_exit_pager_early(self) -> None:
        with isolation.isolate(infinite) as env:
            pager = isolation.PagerControl(env)
            self.assertEqual(MAX_LINES_PER_PAGE, pager.advance())
            self.assertEqual(MAX_LINES_PER_PAGE, pager.quit())
            self.assertFalse(env.error_output())
        self.assertEqual(141, env.exit_code())

    def test_interrupt_early(self) -> None:
        with isolation.isolate(infinite) as env:
            pager = isolation.PagerControl(env)
            self.assertEqual(MAX_LINES_PER_PAGE, pager.advance())
            env.interrupt()
            while pager.advance():
                continue
            pager.quit()
            self.assertGreater(pager.total_lines(), MAX_LINES_PER_PAGE)
            self.assertFalse(env.error_output())
        self.assertEqual(130, env.exit_code())

    def test_interrupt_early_quit(self) -> None:
        with isolation.isolate(infinite) as env:
            pager = isolation.PagerControl(env)
            self.assertEqual(MAX_LINES_PER_PAGE, pager.advance())
            env.interrupt()
            pager.quit()
            self.assertGreater(pager.total_lines(), MAX_LINES_PER_PAGE)
            self.assertFalse(env.error_output())
        self.assertEqual(130, env.exit_code())

    def test_interrupt_in_middle_after_complete(self) -> None:
        num_lines = 100
        with isolation.isolate(finite(num_lines)) as env:
            pager = isolation.PagerControl(env)
            self.assertEqual(MAX_LINES_PER_PAGE, pager.advance())
            for i in range(100):
                env.interrupt()
            self.assertEqual(MAX_LINES_PER_PAGE, pager.quit())
            self.assertFalse(env.error_output())
        self.assertEqual(0, env.exit_code())

    def test_interrupt_at_end_after_complete(self) -> None:
        num_lines = 100
        with isolation.isolate(finite(num_lines)) as env:
            pager = isolation.PagerControl(env)
            while pager.advance():
                continue
            self.assertEqual(num_lines, pager.total_lines())
            for i in range(100):
                env.interrupt()
            self.assertEqual(0, pager.quit())
            self.assertFalse(env.error_output())
        self.assertEqual(0, env.exit_code())

    def test_short_output(self) -> None:
        num_lines = 10
        with isolation.isolate(finite(num_lines)) as env:
            pager = isolation.PagerControl(env)
            for i, l in enumerate(pager.read_lines(num_lines)):
                self.assertEqual(str(i), l.rstrip())
            self.assertFalse(env.error_output())
        self.assertEqual(0, env.exit_code())

    def test_short_output_reset(self) -> None:
        num_lines = 10
        with isolation.isolate(finite(num_lines, reset_on_exit=True)) as env:
            pager = isolation.PagerControl(env)
            self.assertEqual(num_lines, pager.quit())
            self.assertFalse(env.error_output())
        self.assertEqual(0, env.exit_code())

    def test_short_streaming_output(self) -> None:
        num_lines = 10
        with isolation.isolate(from_stdin, stdin_pipe=True) as env:
            pager = isolation.PagerControl(env)
            with env.stdin_pipe() as in_pipe:
                for i in range(num_lines):
                    print(i, file=in_pipe)
            for i, l in enumerate(pager.read_lines(num_lines)):
                self.assertEqual(i, int(l))
            env.interrupt()
            self.assertEqual(0, pager.quit())
            self.assertFalse(env.error_output())
        self.assertEqual(0, env.exit_code())

    def test_exception(self) -> None:
        num_lines = 50
        with isolation.isolate(with_exception) as env:
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
        self.assertEqual(1, env.exit_code())

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