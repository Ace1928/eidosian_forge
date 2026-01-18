import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
def test_printer_print():
    p = Printer()
    text = 'This is a test.'
    p.good(text)
    p.fail(text)
    p.info(text)
    p.text(text)