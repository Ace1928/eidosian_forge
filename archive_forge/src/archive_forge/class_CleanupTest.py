import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
class CleanupTest(unittest.TestCase):

    def test_no_pager_stream_not_closed(self) -> None:
        flush = mock.MagicMock()
        with sinks.BufferFixture() as out:
            with autopage.AutoPager(out.stream) as stream:
                stream.flush = flush
                stream.write('foo')
            self.assertFalse(out.stream.closed)
        flush.assert_called_once()

    def test_no_pager_broken_pipe(self) -> None:
        flush = mock.MagicMock(side_effect=BrokenPipeError)
        with sinks.BufferFixture() as out:
            with autopage.AutoPager(out.stream) as stream:
                stream.flush = flush
                stream.write('foo')
            self.assertTrue(out.stream.closed)
        flush.assert_called_once()

    def test_no_pager_broken_pipe_flush(self) -> None:
        flush = mock.MagicMock(side_effect=BrokenPipeError)
        with sinks.BufferFixture() as out:
            with autopage.AutoPager(out.stream) as stream:
                stream.write('foo')
                stream.flush = flush
            self.assertTrue(out.stream.closed)
        flush.assert_called_once()

    def test_no_pager_stream_closed(self) -> None:
        flush = mock.MagicMock(side_effect=ValueError)
        with sinks.BufferFixture() as out:
            with autopage.AutoPager(out.stream) as stream:
                stream.write('foo')
                stream.close()
                stream.flush = flush
            self.assertTrue(out.stream.closed)

    def test_pager_stream_not_closed(self) -> None:
        with sinks.TTYFixture() as out:
            ap = autopage.AutoPager(out.stream)
            with fixtures.MockPatch('subprocess.Popen') as popen:
                with sinks.BufferFixture() as pager_in:
                    popen.mock.return_value.stdin = pager_in.stream
                    with ap as stream:
                        self.assertIs(pager_in.stream, stream)
                    self.assertTrue(pager_in.stream.closed)

    def test_pager_stream_not_closed_interrupt(self) -> None:
        with sinks.TTYFixture() as out:
            ap = autopage.AutoPager(out.stream)
            with fixtures.MockPatch('subprocess.Popen') as popen:
                with sinks.BufferFixture() as pager_in:
                    popen.mock.return_value.stdin = pager_in.stream

                    def run() -> None:
                        with ap as stream:
                            self.assertIs(pager_in.stream, stream)
                            raise KeyboardInterrupt
                    self.assertRaises(KeyboardInterrupt, run)
                    self.assertTrue(pager_in.stream.closed)

    def test_pager_broken_pipe(self) -> None:
        flush = mock.MagicMock(side_effect=BrokenPipeError)
        with sinks.TTYFixture() as out:
            ap = autopage.AutoPager(out.stream)
            with fixtures.MockPatch('subprocess.Popen') as popen:
                with sinks.BufferFixture() as pager_in:
                    popen.mock.return_value.stdin = pager_in.stream
                    pager_in.stream.flush = flush
                    with ap as stream:
                        self.assertIs(pager_in.stream, stream)
                    self.assertTrue(pager_in.stream.closed)
                    popen.mock.return_value.wait.assert_called_once()

    def test_pager_stream_closed(self) -> None:
        with sinks.TTYFixture() as out:
            ap = autopage.AutoPager(out.stream)
            with fixtures.MockPatch('subprocess.Popen') as popen:
                with sinks.BufferFixture() as pager_in:
                    popen.mock.return_value.stdin = pager_in.stream
                    with ap as stream:
                        self.assertIs(pager_in.stream, stream)
                        stream.close()
                    popen.mock.return_value.wait.assert_called_once()