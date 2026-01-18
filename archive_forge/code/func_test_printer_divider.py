import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
def test_printer_divider():
    p = Printer(line_max=20, no_print=True)
    p.divider() == '\x1b[1m\n================\x1b[0m'
    p.divider('test') == '\x1b[1m\n====== test ======\x1b[0m'
    p.divider('test', char='*') == '\x1b[1m\n****** test ******\x1b[0m'
    assert p.divider('This is a very long text, it is very long') == '\x1b[1m\n This is a very long text, it is very long \x1b[0m'
    with pytest.raises(ValueError):
        p.divider('test', char='~.')