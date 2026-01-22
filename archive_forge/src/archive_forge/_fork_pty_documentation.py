import os
import errno
from pty import (STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO, CHILD)
from .util import PtyProcessError
This makes the pseudo-terminal the controlling tty. This should be
    more portable than the pty.fork() function. Specifically, this should
    work on Solaris. 