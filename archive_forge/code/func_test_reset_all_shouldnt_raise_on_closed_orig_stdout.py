from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def test_reset_all_shouldnt_raise_on_closed_orig_stdout(self):
    stream = StringIO()
    converter = AnsiToWin32(stream)
    stream.close()
    converter.reset_all()