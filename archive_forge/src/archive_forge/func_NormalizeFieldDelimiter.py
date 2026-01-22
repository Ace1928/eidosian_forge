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
def NormalizeFieldDelimiter(field_delimiter: str) -> str:
    """Validates and returns the correct field_delimiter."""
    if field_delimiter is None:
        return field_delimiter
    key = field_delimiter.lower()
    return _DELIMITER_MAP.get(key, field_delimiter)