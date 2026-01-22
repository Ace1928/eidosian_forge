import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
class NoPagerTest(unittest.TestCase):

    def test_pipe_output_to_end(self) -> None:
        num_lines = 100
        with isolation.isolate(finite(num_lines), stdout_pipe=True) as env:
            with env.stdout_pipe() as out:
                output = out.readlines()
            self.assertEqual(num_lines, len(output))
            self.assertFalse(env.error_output())
        self.assertEqual(0, env.exit_code())

    def test_exit_early(self) -> None:
        with isolation.isolate(infinite, stdout_pipe=True) as env:
            with env.stdout_pipe() as out:
                for i in range(500):
                    out.readline()
            self.assertFalse(env.error_output())
        self.assertEqual(141, env.exit_code())

    def test_exit_early_buffered(self) -> None:
        num_lines = 10
        with isolation.isolate(from_stdin, stdin_pipe=True, stdout_pipe=True) as env:
            with env.stdin_pipe() as in_pipe:
                for i in range(num_lines):
                    print(i, file=in_pipe)
            with env.stdout_pipe():
                pass
            self.assertFalse(env.error_output())
        self.assertEqual(141, env.exit_code())

    def test_interrupt_early(self) -> None:
        with isolation.isolate(infinite, stdout_pipe=True) as env:
            env.interrupt()
            with env.stdout_pipe() as out:
                output = out.readlines()
            self.assertGreater(len(output), 0)
            self.assertFalse(env.error_output())
        self.assertEqual(130, env.exit_code())

    def test_short_streaming_output(self) -> None:
        num_lines = 10
        with isolation.isolate(from_stdin, stdin_pipe=True, stdout_pipe=True) as env:
            with env.stdin_pipe() as in_pipe:
                for i in range(num_lines):
                    print(i, file=in_pipe)
            with env.stdout_pipe() as out:
                for i in range(num_lines):
                    self.assertEqual(i, int(out.readline()))
                env.interrupt()
                self.assertEqual(0, len(out.readlines()))
            self.assertFalse(env.error_output())
        self.assertEqual(0, env.exit_code())