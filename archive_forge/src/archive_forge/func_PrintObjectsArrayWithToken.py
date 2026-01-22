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
def PrintObjectsArrayWithToken(object_infos, objects_type):
    if FLAGS.format in ['prettyjson', 'json']:
        bq_utils.PrintFormattedJsonObject(object_infos)
    elif FLAGS.format in [None, 'sparse', 'pretty']:
        PrintObjectsArray(object_infos['results'], objects_type)
        if 'token' in object_infos:
            print('\nNext token: ' + object_infos['token'])