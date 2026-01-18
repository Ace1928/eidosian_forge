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
@tools.attach(view, 'darwin')
def view_darwin(filepath):
    """Open filepath with its default application (mac)."""
    subprocess.Popen(['open', filepath])