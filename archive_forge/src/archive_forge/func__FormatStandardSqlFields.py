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
def _FormatStandardSqlFields(standard_sql_fields):
    """Returns a string with standard_sql_fields.

  Currently only supports printing primitive field types and repeated fields.
  Args:
    standard_sql_fields: A list of standard sql fields.

  Returns:
    The formatted standard sql fields.
  """
    lines = []
    for field in standard_sql_fields:
        if field['type']['typeKind'] == 'ARRAY':
            field_type = field['type']['arrayElementType']['typeKind']
        else:
            field_type = field['type']['typeKind']
        entry = '|- %s: %s' % (field['name'], field_type.lower())
        if field['type']['typeKind'] == 'ARRAY':
            entry += ' (repeated)'
        lines.extend([entry])
    return '\n'.join(lines)