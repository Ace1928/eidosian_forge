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
def ParseParameterType(type_string):
    """Parse a parameter type string into a JSON dict for the BigQuery API."""
    type_dict = {'type': type_string.upper()}
    if type_string.upper().startswith('ARRAY<') and type_string.endswith('>'):
        type_dict = {'type': 'ARRAY', 'arrayType': ParseParameterType(type_string[6:-1])}
    if type_string.startswith('STRUCT<') and type_string.endswith('>'):
        type_dict = {'type': 'STRUCT', 'structTypes': ParseStructType(type_string[7:-1])}
    if type_string.startswith('RANGE<') and type_string.endswith('>'):
        type_dict = {'type': 'RANGE', 'rangeElementType': ParseParameterType(type_string[6:-1])}
    if not type_string:
        raise app.UsageError('Query parameter missing type')
    return type_dict