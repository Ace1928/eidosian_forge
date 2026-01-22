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
def RunDebug(self, args, kwds):
    """Run this command in debug mode."""
    logging.debug('In NewCmd.RunDebug: %s, %s', args, kwds)
    try:
        return_value = self.RunWithArgs(*args, **kwds)
    except (BaseException, googleapiclient.errors.ResumableUploadError) as e:
        if isinstance(e, app.UsageError) or (isinstance(e, bq_error.BigqueryError) and (not isinstance(e, bq_error.BigqueryInterfaceError))) or isinstance(e, googleapiclient.errors.ResumableUploadError):
            return self._HandleError(e)
        print()
        print('****************************************************')
        print('**  Unexpected Exception raised in bq execution!  **')
        if FLAGS.headless:
            print('**  --headless mode enabled, exiting.             **')
            print('**  See STDERR for traceback.                     **')
        else:
            print('**  --debug_mode enabled, starting pdb.           **')
        print('****************************************************')
        print()
        traceback.print_exc()
        print()
        if not FLAGS.headless:
            pdb.post_mortem()
        return 1
    return return_value