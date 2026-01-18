import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
def test_printer_custom():
    colors = {'yellow': 220, 'purple': 99}
    icons = {'warn': '⚠️', 'question': '?'}
    p = Printer(no_print=True, colors=colors, icons=icons)
    text = 'This is a test.'
    purple_question = p.text(text, color='purple', icon='question')
    warning = p.warn(text)
    if SUPPORTS_ANSI and (not NO_UTF8):
        assert purple_question == '\x1b[38;5;99m? {}\x1b[0m'.format(text)
        assert warning == '\x1b[38;5;3m⚠️ {}\x1b[0m'.format(text)
    if SUPPORTS_ANSI and NO_UTF8:
        assert purple_question == '\x1b[38;5;99m? {}\x1b[0m'.format(text)
        assert warning == '\x1b[38;5;3m?? {}\x1b[0m'.format(text)
    if not SUPPORTS_ANSI and (not NO_UTF8):
        assert purple_question == '? {}'.format(text)
        assert warning == '⚠️ {}'.format(text)
    if not SUPPORTS_ANSI and NO_UTF8:
        assert purple_question == '? {}'.format(text)
        assert warning == '?? {}'.format(text)