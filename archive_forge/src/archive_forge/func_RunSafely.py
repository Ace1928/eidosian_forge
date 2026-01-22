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
def RunSafely(self, args, kwds):
    """Run this command, printing information about any exceptions raised."""
    logging.debug('In BigqueryCmd.RunSafely: %s, %s', args, kwds)
    try:
        return_value = self.RunWithArgs(*args, **kwds)
    except BaseException as e:
        return bq_utils.ProcessError(e, name=self._command_name)
    return return_value