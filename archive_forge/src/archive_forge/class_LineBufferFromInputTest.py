import unittest
import fixtures  # type: ignore
from autopage.tests import sinks
import autopage
class LineBufferFromInputTest(unittest.TestCase):

    def test_tty(self) -> None:
        with sinks.TTYFixture() as inp:
            self.assertTrue(autopage.line_buffer_from_input(inp.stream))

    def test_file(self) -> None:
        with sinks.TempFixture() as inp:
            self.assertFalse(autopage.line_buffer_from_input(inp.stream))

    def test_default_tty(self) -> None:
        with sinks.TTYFixture() as inp:
            with fixtures.MonkeyPatch('sys.stdin', inp.stream):
                self.assertTrue(autopage.line_buffer_from_input())

    def test_default_file(self) -> None:
        with sinks.TempFixture() as inp:
            with fixtures.MonkeyPatch('sys.stdin', inp.stream):
                self.assertFalse(autopage.line_buffer_from_input())