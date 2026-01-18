import sys
from unittest import TestCase, main
from ..ansitowin32 import StreamWrapper, AnsiToWin32
from .utils import pycharm, replace_by, replace_original_by, StreamTTY, StreamNonTTY
def test_withPycharmStreamWrapped(self):
    with pycharm():
        self.assertTrue(AnsiToWin32(StreamTTY()).stream.isatty())
        self.assertFalse(AnsiToWin32(StreamNonTTY()).stream.isatty())
        self.assertTrue(AnsiToWin32(sys.stdout).stream.isatty())
        self.assertTrue(AnsiToWin32(sys.stderr).stream.isatty())