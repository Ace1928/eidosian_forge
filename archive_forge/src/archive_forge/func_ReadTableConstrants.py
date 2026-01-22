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
def ReadTableConstrants(table_constraints: str):
    """Create table constraints json object from string or a file name.

  Args:
    table_constraints: Either a json string that presents a table_constraints
      proto or name of a file that contains the json string.

  Returns:
    The table_constraints (as a json object).

  Raises:
    bq_error.BigqueryTableConstraintsError: If load the table constraints
      from the string or file failed.
  """
    if not table_constraints:
        raise bq_error.BigqueryTableConstraintsError('table_constraints cannot be empty')
    if os.path.exists(table_constraints):
        with open(table_constraints) as f:
            try:
                loaded_json = json.load(f)
            except ValueError as e:
                raise bq_error.BigqueryTableConstraintsError('Error decoding JSON table constraints from file %s.' % (table_constraints,)) from e
        return loaded_json
    if re.search('^[./~\\\\]', table_constraints) is not None:
        raise bq_error.BigqueryTableConstraintsError('Error reading table constraints: "%s" looks like a filename, but was not found.' % (table_constraints,))
    try:
        loaded_json = json.loads(table_constraints)
    except ValueError as e:
        raise bq_error.BigqueryTableConstraintsError('Error decoding JSON table constraints from string %s.' % (table_constraints,)) from e
    return loaded_json