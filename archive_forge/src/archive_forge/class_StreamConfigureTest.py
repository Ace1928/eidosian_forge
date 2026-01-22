import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
class StreamConfigureTest(fixtures.TestWithFixtures):

    def setUp(self) -> None:
        out = sinks.TempFixture()
        self.useFixture(out)
        self.stream = out.stream
        self.default_lb = self.stream.line_buffering
        self.default_errors = self.stream.errors
        self.encoding = self.stream.encoding

    def test_line_buffering_on(self) -> None:
        ap = autopage.AutoPager(self.stream, line_buffering=True)
        ap._reconfigure_output_stream()
        self.addCleanup(ap._out.close)
        self.assertTrue(ap._out.line_buffering)
        self.assertEqual(self.default_errors, ap._out.errors)
        self.assertEqual(self.encoding, ap._out.encoding)
        self.assertIs(True, ap._line_buffering())
        self.assertEqual(self.default_errors, ap._errors())

    def test_line_buffering_off(self) -> None:
        ap = autopage.AutoPager(self.stream, line_buffering=False)
        ap._reconfigure_output_stream()
        self.addCleanup(ap._out.close)
        self.assertFalse(ap._out.line_buffering)
        self.assertEqual(self.default_errors, ap._out.errors)
        self.assertEqual(self.encoding, ap._out.encoding)
        self.assertIs(False, ap._line_buffering())
        self.assertEqual(self.default_errors, ap._errors())

    def test_stdout_line_buffering_on(self) -> None:
        with fixtures.MonkeyPatch('sys.stdout', self.stream):
            ap = autopage.AutoPager(line_buffering=True)
            ap._reconfigure_output_stream()
            self.addCleanup(ap._out.close)
            self.assertTrue(sys.stdout.line_buffering)
            self.assertEqual(self.default_errors, sys.stdout.errors)
            self.assertEqual(self.encoding, sys.stdout.encoding)

    def test_errors(self) -> None:
        ap = autopage.AutoPager(self.stream, errors=autopage.ErrorStrategy.NAME_REPLACE)
        ap._reconfigure_output_stream()
        self.addCleanup(ap._out.close)
        self.assertEqual(self.default_lb, ap._out.line_buffering)
        self.assertEqual('namereplace', ap._out.errors)
        self.assertNotEqual(self.default_errors, ap._out.errors)
        self.assertEqual(self.encoding, ap._out.encoding)
        self.assertEqual('namereplace', ap._errors())
        self.assertEqual(self.default_lb, ap._line_buffering())

    def test_errors_string(self) -> None:
        ap = autopage.AutoPager(self.stream, errors='namereplace')
        ap._reconfigure_output_stream()
        self.addCleanup(ap._out.close)
        self.assertEqual(self.default_lb, ap._out.line_buffering)
        self.assertEqual('namereplace', ap._out.errors)
        self.assertNotEqual(self.default_errors, ap._out.errors)
        self.assertEqual(self.encoding, ap._out.encoding)
        self.assertEqual('namereplace', ap._errors())
        self.assertEqual(self.default_lb, ap._line_buffering())

    def test_errors_bogus_string(self) -> None:
        self.assertRaises(ValueError, autopage.AutoPager, self.stream, errors='panic')

    def test_line_buffering_on_errors(self) -> None:
        ap = autopage.AutoPager(self.stream, line_buffering=True, errors=autopage.ErrorStrategy.NAME_REPLACE)
        ap._reconfigure_output_stream()
        self.addCleanup(ap._out.close)
        self.assertTrue(ap._out.line_buffering)
        self.assertEqual('namereplace', ap._out.errors)
        self.assertNotEqual(self.default_errors, ap._out.errors)
        self.assertEqual(self.encoding, ap._out.encoding)
        self.assertIs(True, ap._line_buffering())
        self.assertEqual('namereplace', ap._errors())