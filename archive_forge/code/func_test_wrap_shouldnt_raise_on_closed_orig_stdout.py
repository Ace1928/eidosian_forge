from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def test_wrap_shouldnt_raise_on_closed_orig_stdout(self):
    stream = StringIO()
    stream.close()
    with patch('colorama.ansitowin32.os.name', 'nt'), patch('colorama.ansitowin32.winapi_test', lambda: True):
        converter = AnsiToWin32(stream)
    self.assertTrue(converter.strip)
    self.assertFalse(converter.convert)