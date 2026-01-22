import os
import sys
import warnings
from shutil import get_terminal_size as _get_terminal_size
def _set_term_title_xterm(title):
    """ Change virtual terminal title in xterm-workalikes """
    global _xterm_term_title_saved
    if not _xterm_term_title_saved:
        sys.stdout.write('\x1b[22;0t')
        _xterm_term_title_saved = True
    sys.stdout.write('\x1b]0;%s\x07' % title)