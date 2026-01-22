from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import datetime
import hashlib
import json
import logging
import os
import random
import re
import string
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
from absl import flags
import googleapiclient
import httplib2
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def _OverwriteCurrentLine(s: str, previous_token=None) -> int:
    """Print string over the current terminal line, and stay on that line.

  The full width of any previous output (by the token) will be wiped clean.
  If multiple callers call this at the same time, it would be bad.

  Args:
    s: string to print.  May not contain newlines.
    previous_token: token returned from previous call, or None on first call.

  Returns:
    a token to pass into your next call to this function.
  """
    if previous_token is not None:
        sys.stderr.write('\r' + ' ' * previous_token)
    sys.stderr.write('\r' + s)
    sys.stderr.flush()
    return len(s)