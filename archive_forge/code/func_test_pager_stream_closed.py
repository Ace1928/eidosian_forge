import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
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