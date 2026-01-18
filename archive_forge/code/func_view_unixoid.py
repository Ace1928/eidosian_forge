import os
import io
import re
import sys
import errno
import platform
import subprocess
import contextlib
from ._compat import stderr_write_binary
from . import tools
@tools.attach(view, 'linux')
@tools.attach(view, 'freebsd')
def view_unixoid(filepath):
    """Open filepath in the user's preferred application (linux, freebsd)."""
    subprocess.Popen(['xdg-open', filepath])