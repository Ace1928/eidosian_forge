import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_stdout_line_buffering_on(self) -> None:
    with fixtures.MonkeyPatch('sys.stdout', self.stream):
        ap = autopage.AutoPager(line_buffering=True)
        ap._reconfigure_output_stream()
        self.addCleanup(ap._out.close)
        self.assertTrue(sys.stdout.line_buffering)
        self.assertEqual(self.default_errors, sys.stdout.errors)
        self.assertEqual(self.encoding, sys.stdout.encoding)