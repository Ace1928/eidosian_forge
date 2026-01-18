import re
import sys
import os
from .ansi import AnsiFore, AnsiBack, AnsiStyle, Style, BEL
from .winterm import enable_vt_processing, WinTerm, WinColor, WinStyle
from .win32 import windll, winapi_test
def write_plain_text(self, text, start, end):
    if start < end:
        self.wrapped.write(text[start:end])
        self.wrapped.flush()