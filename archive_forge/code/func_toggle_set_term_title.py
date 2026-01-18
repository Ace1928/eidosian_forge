import os
import sys
import warnings
from shutil import get_terminal_size as _get_terminal_size
def toggle_set_term_title(val):
    """Control whether set_term_title is active or not.

    set_term_title() allows writing to the console titlebar.  In embedded
    widgets this can cause problems, so this call can be used to toggle it on
    or off as needed.

    The default state of the module is for the function to be disabled.

    Parameters
    ----------
    val : bool
        If True, set_term_title() actually writes to the terminal (using the
        appropriate platform-specific module).  If False, it is a no-op.
    """
    global ignore_termtitle
    ignore_termtitle = not val