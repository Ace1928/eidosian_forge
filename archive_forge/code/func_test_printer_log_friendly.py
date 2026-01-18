import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
def test_printer_log_friendly():
    text = 'This is a test.'
    ENV_LOG_FRIENDLY = 'WASABI_LOG_FRIENDLY'
    os.environ[ENV_LOG_FRIENDLY] = 'True'
    p = Printer(no_print=True)
    assert p.good(text) in ('✔ This is a test.', '[+] This is a test.')
    del os.environ[ENV_LOG_FRIENDLY]