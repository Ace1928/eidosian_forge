from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import datetime
import functools
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from absl import app
from absl import flags
import yaml
import table_formatter
import bq_utils
from clients import utils as bq_client_utils
from utils import bq_error
from utils import bq_id_utils
from pyglib import stringutil
def ParseFileSetSpecType(file_set_spec_type=None):
    """Parses the file set specification type from the arguments.

  Args:
    file_set_spec_type: specifies how to discover files given source URIs.

  Returns:
    file set specification type.
  Raises:
    UsageError: when an illegal value is passed.
  """
    if file_set_spec_type is None:
        return None
    valid_spec_types = ['FILE_SYSTEM_MATCH', 'NEW_LINE_DELIMITED_MANIFEST']
    if file_set_spec_type not in valid_spec_types:
        raise app.UsageError('Error parsing file_set_spec_type, only FILE_SYSTEM_MATCH, NEW_LINE_DELIMITED_MANIFEST or no value are accepted')
    return 'FILE_SET_SPEC_TYPE_' + file_set_spec_type