import traceback
import pytest
from wasabi.traceback_printer import TracebackPrinter
def test_traceback_printer_only_title(tb):
    tbp = TracebackPrinter(tb_base='wasabi')
    msg = tbp('Hello world', tb=tb)
    print(msg)