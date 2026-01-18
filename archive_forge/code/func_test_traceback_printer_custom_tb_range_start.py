import traceback
import pytest
from wasabi.traceback_printer import TracebackPrinter
def test_traceback_printer_custom_tb_range_start():
    tbp = TracebackPrinter(tb_range_start=-1)
    msg = tbp('Hello world', 'This is a test')
    print(msg)