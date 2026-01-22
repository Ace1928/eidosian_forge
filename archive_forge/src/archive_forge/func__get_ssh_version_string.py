import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
def _get_ssh_version_string(self, args):
    """Return SSH version string from the subprocess."""
    try:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, **os_specific_subprocess_params())
        stdout, stderr = p.communicate()
    except OSError:
        stdout = stderr = b''
    return (stdout + stderr).decode(osutils.get_terminal_encoding())