import errno
import os
import pdb
import socket
import stat
import struct
import sys
import time
import traceback
import gflags as flags
class HelpshortFlag(flags.BooleanFlag):
    """Special bool flag that calls usage(shorthelp=1) and raises SystemExit."""

    def __init__(self):
        flags.BooleanFlag.__init__(self, 'helpshort', 0, 'show usage only for this module', allow_override=1)

    def Parse(self, arg):
        if arg:
            usage(shorthelp=1, writeto_stdout=1)
            sys.exit(1)