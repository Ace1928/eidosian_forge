import os.path
import signal
import sys
import pexpect
def zsh(command='zsh', args=('--no-rcs', '-V', '+Z')):
    """Start a zsh shell and return a :class:`REPLWrapper` object."""
    return _repl_sh(command, list(args), non_printable_insert='%(!..)')