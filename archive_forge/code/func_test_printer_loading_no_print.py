import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
def test_printer_loading_no_print():
    p = Printer(no_print=True)
    with p.loading('Loading...'):
        time.sleep(1)
    p.good('Success!')