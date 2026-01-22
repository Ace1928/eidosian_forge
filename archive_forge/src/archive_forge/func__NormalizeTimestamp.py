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
def _NormalizeTimestamp(unused_field, value):
    """Returns bq-specific formatting of a TIMESTAMP type."""
    try:
        date = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc) + datetime.timedelta(seconds=float(value))
        date = date.replace(tzinfo=None)
        date = date.replace(microsecond=0)
        return date.isoformat(' ')
    except ValueError:
        return '<date out of range for display>'