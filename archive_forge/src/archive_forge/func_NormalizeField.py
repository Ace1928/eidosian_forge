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
@staticmethod
def NormalizeField(field, value):
    """Returns bq-specific formatting of a field."""
    if value is None:
        return None
    normalizer = TablePrinter._FIELD_NORMALIZERS.get(field.get('type', '').upper(), lambda _, x: x)
    if field.get('mode', '').upper() == 'REPEATED':
        return [normalizer(field, value) for value in value]
    return normalizer(field, value)