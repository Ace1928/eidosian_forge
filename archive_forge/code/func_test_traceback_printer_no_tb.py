import traceback
import pytest
from wasabi.traceback_printer import TracebackPrinter
def test_traceback_printer_no_tb():
    tbp = TracebackPrinter(tb_base='wasabi')
    msg = tbp('Hello world', 'This is a test')
    print(msg)