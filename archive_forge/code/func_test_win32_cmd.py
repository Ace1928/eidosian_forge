import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_win32_cmd(self) -> None:
    with PlatformFixture('win32'):
        cmd = command.PlatformPager()
        self.assertEqual(['more.com'], cmd.command())