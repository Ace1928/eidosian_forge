import traceback
import pytest
from wasabi.traceback_printer import TracebackPrinter
def test_traceback_printer_highlight(tb):
    tbp = TracebackPrinter(tb_base='wasabi')
    msg = tbp('Hello world', 'This is a test', tb=tb, highlight='kwargs')
    print(msg)