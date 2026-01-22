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
def _get_vendor_specific_argv(self, username, host, port, subsystem=None, command=None):
    self._check_hostname(host)
    args = [self.executable_path, '-x', '-a', '-ssh', '-2', '-batch']
    if port is not None:
        args.extend(['-P', str(port)])
    if username is not None:
        args.extend(['-l', username])
    if subsystem is not None:
        args.extend(['-s', host, subsystem])
    else:
        args.extend([host] + command)
    return args