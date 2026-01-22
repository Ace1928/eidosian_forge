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
def RegisterAndParseFlagsWithUsage():
    """Register help flags, parse arguments and show usage if appropriate.

  Returns:
    remaining arguments after flags parsing
  """
    DefineHelpFlags()
    argv = parse_flags_with_usage(sys.argv)
    return argv