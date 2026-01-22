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
class HelpFlag(flags.BooleanFlag):
    """Special boolean flag that displays usage and raises SystemExit."""

    def __init__(self):
        flags.BooleanFlag.__init__(self, 'help', 0, 'show this help', short_name='?', allow_override=1)

    def Parse(self, arg):
        if arg:
            usage(writeto_stdout=1)
            sys.exit(1)