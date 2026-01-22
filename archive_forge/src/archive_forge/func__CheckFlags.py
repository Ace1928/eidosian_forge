import logging
import os
import pdb
import shlex
import sys
import traceback
import types
from absl import app
from absl import flags
import googleapiclient
import bq_flags
import bq_utils
from utils import bq_error
from utils import bq_logging
from utils import bq_processor_utils
from pyglib import appcommands
def _CheckFlags(self):
    """Validate flags after command specific flags have been loaded.

    This function will run through all values in appcommands._cmd_argv and
    pick out any unused flags and verify their validity.  If the flag is
    not defined, we will print the flags.FlagsError text and exit; otherwise,
    we will print a positioning error message and exit.  Print statements
    were used in this case because raising app.UsageError caused the usage
    help text to be printed.

    If no extraneous flags exist, this function will do nothing.
    """
    unused_flags = [f for f in appcommands.GetCommandArgv() if f.startswith('--') or f.startswith('-')]
    for flag in unused_flags:
        flag_name = flag[4:] if flag.startswith('--no') else flag[2:]
        flag_name = flag_name.split('=')[0]
        if flag_name not in FLAGS:
            print("FATAL Flags parsing error: Unknown command line flag '%s'\nRun 'bq help' to get help" % flag)
            sys.exit(1)
        else:
            print("FATAL Flags positioning error: Flag '%s' appears after final command line argument. Please reposition the flag.\nRun 'bq help' to get help." % flag)
            sys.exit(1)