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
class BuildDataFlag(flags.BooleanFlag):
    """Boolean flag that writes build data to stdout and exits."""

    def __init__(self):
        flags.BooleanFlag.__init__(self, 'show_build_data', 0, 'show build data and exit')

    def Parse(self, arg):
        if arg:
            sys.stdout.write(build_data.BuildData())
            sys.exit(0)