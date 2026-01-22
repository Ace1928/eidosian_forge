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
def ParseStructType(type_string):
    """Parse a Struct QueryParameter type into a JSON dict form."""
    subtypes = []
    for name, sub_type in StructTypeSplit(type_string):
        subtypes.append({'type': ParseParameterType(sub_type), 'name': name})
    return subtypes