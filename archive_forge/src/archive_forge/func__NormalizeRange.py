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
def _NormalizeRange(field, value):
    """Returns bq-specific formatting of a RANGE type."""
    parsed = ParseRangeString(value)
    if parsed is None:
        return '<invalid range>'
    start, end = parsed
    if field.get('rangeElementType').get('type').upper() != 'TIMESTAMP':
        start = start.upper() if IsRangeBoundaryUnbounded(start) else start
        end = end.upper() if IsRangeBoundaryUnbounded(end) else end
        return '[%s, %s)' % (start, end)
    normalized_start = start.upper() if IsRangeBoundaryUnbounded(start) else TablePrinter._NormalizeTimestamp(field, start)
    normalized_end = end.upper() if IsRangeBoundaryUnbounded(end) else TablePrinter._NormalizeTimestamp(field, end)
    return '[%s, %s)' % (normalized_start, normalized_end)