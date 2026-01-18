import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
def test_printer_log_friendly_prefix():
    text = 'This is a test.'
    ENV_LOG_FRIENDLY = 'CUSTOM_LOG_FRIENDLY'
    os.environ[ENV_LOG_FRIENDLY] = 'True'
    p = Printer(no_print=True, env_prefix='CUSTOM')
    assert p.good(text) in ('✔ This is a test.', '[+] This is a test.')
    print(p.good(text))
    del os.environ[ENV_LOG_FRIENDLY]