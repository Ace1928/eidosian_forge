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
Constructor.

        :param proc: a subprocess.Popen
        :param sock: if proc.stdin/out is a socket from a socketpair, then sock
            should breezy's half of that socketpair.  If not passed, proc's
            stdin/out is assumed to be ordinary pipes.
        