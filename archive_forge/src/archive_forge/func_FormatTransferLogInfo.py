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
def FormatTransferLogInfo(transfer_log_info):
    """Prepare transfer log info for printing.

  Arguments:
    transfer_log_info: transfer log info to format.

  Returns:
    The new transfer config log.
  """
    result = {}
    for key, value in transfer_log_info.items():
        result[key] = value
    return result