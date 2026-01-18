import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
def test_printer_no_pretty():
    p = Printer(no_print=True, pretty=False)
    text = 'This is a test.'
    assert p.good(text) == text
    assert p.fail(text) == text
    assert p.warn(text) == text
    assert p.info(text) == text
    assert p.text(text) == text