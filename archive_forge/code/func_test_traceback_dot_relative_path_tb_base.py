import traceback
import pytest
from wasabi.traceback_printer import TracebackPrinter
def test_traceback_dot_relative_path_tb_base(tb):
    tbp = TracebackPrinter(tb_base='.')
    msg = tbp('Hello world', tb=tb)
    print(msg)