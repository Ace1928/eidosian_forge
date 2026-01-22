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
def GetInfoString() -> str:
    """Gets the info string for the current execution."""
    platform_str = GetPlatformString()
    try:
        httplib2_version = httplib2.__version__
    except AttributeError:
        httplib2_version = httplib2.python3.__version__
    try:
        shell_path = os.environ['PATH']
    except KeyError:
        shell_path = None
    try:
        python_path = os.environ['PYTHONPATH']
    except KeyError:
        python_path = None
    return textwrap.dedent('      BigQuery CLI [{version}]\n\n      Platform: [{platform_str}] {uname}\n      Python Version: [{python_version}]\n\n      Requests Version: [{requests_version}]\n      Urllib3 Version: [{urllib3_version}]\n      Httplib2: [{httplib2_version}]\n      Google Auth Version: [{google_auth_version}]\n\n      System PATH: [{sys_path}]\n      Shell PATH: [{shell_path}]\n      Python PATH: [{python_path}]\n\n      '.format(version=VERSION_NUMBER, platform_str=platform_str, uname=platform.uname(), python_version=sys.version.replace('\n', ' '), httplib2_version=httplib2_version, google_auth_version=google_auth_version.__version__, requests_version=requests.__version__, urllib3_version=urllib3.__version__, sys_path=os.pathsep.join(sys.path), shell_path=shell_path, python_path=python_path))