import unittest
import fixtures  # type: ignore
from autopage.tests import sinks
import autopage
def test_tty(self) -> None:
    with sinks.TTYFixture() as inp:
        self.assertTrue(autopage.line_buffer_from_input(inp.stream))