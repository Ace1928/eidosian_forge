import sys
from unittest import TestCase, main
from ..ansitowin32 import StreamWrapper, AnsiToWin32
from .utils import pycharm, replace_by, replace_original_by, StreamTTY, StreamNonTTY
def test_withPycharmNoneOverride(self):
    with pycharm():
        with replace_by(None), replace_original_by(None):
            self.assertFalse(is_a_tty(None))
            self.assertFalse(is_a_tty(StreamNonTTY()))
            self.assertTrue(is_a_tty(StreamTTY()))