from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import os
import sys
import textwrap
import time
from typing import Optional, TextIO
from absl import app
from absl import flags
import termcolor
import bq_flags
import bq_utils
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import utils as frontend_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_logging
from utils import bq_processor_utils
from pyglib import stringutil
def DeleteCredentials(self) -> Optional[int]:
    """Deletes this user's credential file."""
    bq_utils.ProcessBigqueryrc()
    filename = FLAGS.service_account_credential_file or FLAGS.credential_file
    if not os.path.exists(filename):
        print('Credential file %s does not exist.' % (filename,))
        return 0
    try:
        if 'y' != frontend_utils.PromptYN('Delete credential file %s? (y/N) ' % (filename,)):
            print('NOT deleting %s, exiting.' % (filename,))
            return 0
        os.remove(filename)
    except OSError as e:
        print('Error removing %s: %s' % (filename, e))
        return 1