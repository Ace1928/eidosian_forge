import traceback
import pytest
from wasabi.traceback_printer import TracebackPrinter
def test_traceback_printer_custom_tb_range():
    tbp = TracebackPrinter(tb_range_start=-10, tb_range_end=-3)
    msg = tbp('Hello world', 'This is a test')
    print(msg)