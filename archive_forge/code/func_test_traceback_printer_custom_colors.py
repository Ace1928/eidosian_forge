import traceback
import pytest
from wasabi.traceback_printer import TracebackPrinter
def test_traceback_printer_custom_colors(tb):
    tbp = TracebackPrinter(tb_base='wasabi', color_error='blue', color_highlight='green', color_tb='yellow')
    msg = tbp('Hello world', 'This is a test', tb=tb, highlight='kwargs')
    print(msg)