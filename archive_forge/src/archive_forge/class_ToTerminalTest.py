import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
class ToTerminalTest(unittest.TestCase):

    def test_pty(self) -> None:
        with sinks.TTYFixture() as out:
            ap = autopage.AutoPager(out.stream)
            self.assertTrue(ap.to_terminal())

    def test_stringio(self) -> None:
        with sinks.BufferFixture() as out:
            ap = autopage.AutoPager(out.stream)
            self.assertFalse(ap.to_terminal())

    def test_file(self) -> None:
        with sinks.TempFixture() as out:
            ap = autopage.AutoPager(out.stream)
            self.assertFalse(ap.to_terminal())

    def test_default_pty(self) -> None:
        with sinks.TTYFixture() as out:
            with fixtures.MonkeyPatch('sys.stdout', out.stream):
                ap = autopage.AutoPager()
            self.assertTrue(ap.to_terminal())

    def test_default_file(self) -> None:
        with sinks.TempFixture() as out:
            with fixtures.MonkeyPatch('sys.stdout', out.stream):
                ap = autopage.AutoPager()
            self.assertFalse(ap.to_terminal())

    def test_launch_pager(self) -> None:
        ap = autopage.AutoPager()
        with mock.patch.object(ap, 'to_terminal', return_value=True), mock.patch.object(ap, '_paged_stream') as page, mock.patch.object(ap, '_reconfigure_output_stream') as reconf:
            with ap as stream:
                page.assert_called_once()
                self.assertIs(page.return_value, stream)
                reconf.assert_not_called()

    def test_launch_pager_fail(self) -> None:
        outstream = mock.Mock()
        ap = autopage.AutoPager(outstream)
        with mock.patch.object(ap, 'to_terminal', return_value=True), mock.patch.object(ap, '_paged_stream', side_effect=OSError) as page, mock.patch.object(ap, '_reconfigure_output_stream') as reconf:
            with ap as stream:
                page.assert_called_once()
                reconf.assert_called_once()
                self.assertIs(outstream, stream)

    def test_no_pager(self) -> None:
        outstream = mock.Mock()
        ap = autopage.AutoPager(outstream)
        with mock.patch.object(ap, 'to_terminal', return_value=False), mock.patch.object(ap, '_paged_stream') as page, mock.patch.object(ap, '_reconfigure_output_stream') as reconf:
            with ap as stream:
                page.assert_not_called()
                self.assertIs(outstream, stream)
                reconf.assert_called_once()

    def test_pager_cat(self) -> None:
        outstream = mock.Mock()
        cat = command.CustomPager('cat')
        ap = autopage.AutoPager(outstream, pager_command=cat)
        with mock.patch.object(ap, 'to_terminal', return_value=True), mock.patch.object(ap, '_paged_stream') as page, mock.patch.object(ap, '_reconfigure_output_stream') as reconf:
            with ap as stream:
                page.assert_not_called()
                self.assertIs(outstream, stream)
                reconf.assert_called_once()