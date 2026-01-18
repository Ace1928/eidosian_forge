import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
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