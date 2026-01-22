import os
import sys
import warnings
from shutil import get_terminal_size as _get_terminal_size
def _restore_term_title_xterm():
    global _xterm_term_title_saved
    assert _xterm_term_title_saved
    sys.stdout.write('\x1b[23;0t')
    _xterm_term_title_saved = False