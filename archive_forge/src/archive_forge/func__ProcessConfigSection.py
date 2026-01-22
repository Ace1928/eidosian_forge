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
def _ProcessConfigSection(filename: str, section_name: Optional[str]=None) -> Dict[str, str]:
    """Read a configuration file section returned as a dictionary.

  Args:
    filename: The filename of the configuration file.
    section_name: if None, read the global flag settings.

  Returns:
    A dictionary of flag names and values from that section of the file.
  """
    dictionary = {}
    if not os.path.exists(filename):
        return dictionary
    try:
        with open(filename) as rcfile:
            dictionary = _ProcessSingleConfigSection(rcfile, section_name)
    except IOError:
        pass
    return dictionary