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
def ParseTagKeys(tag_keys: str) -> List[str]:
    """Parses user-supplied string representing tag keys.

  Args:
    tag_keys: A comma separated user-supplied string representing tag keys.  It
      is expected to be in the format "key1,key2".

  Returns:
    A list of tag keys.

  Raises:
    UsageError: Incorrect tag_keys or no tag_keys are supplied.
  """
    tag_keys = tag_keys.strip()
    if not tag_keys:
        raise app.UsageError('No tag keys supplied')
    tags_set = set()
    for key in tag_keys.split(','):
        key = key.strip()
        if not key:
            raise app.UsageError('Tag key cannot be None')
        if key in tags_set:
            raise app.UsageError('Cannot specify tag key "%s" multiple times' % key)
        if key.find(':') != -1:
            raise app.UsageError('Specify only tag key for "%s"' % key)
        tags_set.add(key)
    return list(tags_set)