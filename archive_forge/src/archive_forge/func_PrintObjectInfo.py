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
def PrintObjectInfo(object_info, reference, custom_format, print_reference=True):
    """Prints the object with various formats."""
    if custom_format == 'schema':
        if 'schema' not in object_info or 'fields' not in object_info['schema']:
            raise app.UsageError('Unable to retrieve schema from specified table.')
        bq_utils.PrintFormattedJsonObject(object_info['schema']['fields'])
    elif FLAGS.format in ['prettyjson', 'json']:
        bq_utils.PrintFormattedJsonObject(object_info)
    elif FLAGS.format in [None, 'sparse', 'pretty']:
        formatter = GetFormatterFromFlags()
        bq_client_utils.ConfigureFormatter(formatter, type(reference), print_format=custom_format, object_info=object_info)
        object_info = bq_client_utils.FormatInfoByType(object_info, type(reference))
        if object_info:
            formatter.AddDict(object_info)
        if reference.typename and print_reference:
            print('%s %s\n' % (reference.typename.capitalize(), reference))
        formatter.Print()
        print()
        if isinstance(reference, bq_id_utils.ApiClientHelper.JobReference):
            PrintJobMessages(object_info)
    else:
        formatter = GetFormatterFromFlags()
        formatter.AddColumns(list(object_info.keys()))
        formatter.AddDict(object_info)
        formatter.Print()