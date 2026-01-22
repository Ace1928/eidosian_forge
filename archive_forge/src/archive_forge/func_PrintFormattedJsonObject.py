import codecs
import copy
import http.client
import json
import logging
import os
import pkgutil
import platform
import sys
import textwrap
import time
import traceback
from typing import Any, Dict, List, Optional, TextIO
from absl import app
from absl import flags
from google.auth import version as google_auth_version
from google.oauth2 import credentials as google_oauth2
import googleapiclient
import httplib2
import oauth2client_4_0.client
import requests
import urllib3
from utils import bq_error
from utils import bq_logging
from pyglib import stringutil
def PrintFormattedJsonObject(obj, default_format='json'):
    """Prints obj in a JSON format according to the "--format" flag.

  Args:
    obj: The object to print.
    default_format: The format to use if the "--format" flag does not specify a
      valid json format: 'json' or 'prettyjson'.
  """
    json_formats = ['json', 'prettyjson']
    if FLAGS.format in json_formats:
        use_format = FLAGS.format
    else:
        use_format = default_format
    if use_format == 'json':
        print(json.dumps(obj, separators=(',', ':')))
    elif use_format == 'prettyjson':
        print(json.dumps(obj, sort_keys=True, indent=2))
    else:
        raise ValueError("Invalid json format for printing: '%s', expected one of: %s" % (use_format, json_formats))