from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
from collections import defaultdict
from functools import wraps
import logging
import os
import pickle
import platform
import re
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
import six
from six.moves import input
from six.moves import urllib
import boto
from gslib import VERSION
from gslib.metrics_tuple import Metric
from gslib.utils import system_util
from gslib.utils.unit_util import CalculateThroughput
from gslib.utils.unit_util import HumanReadableToBytes
@classmethod
def _CheckAndSetDisabledCache(cls):
    """Sets _disabled_cache based on user opt-in or out."""
    if os.environ.get('GSUTIL_TEST_ANALYTICS') == '1':
        cls._disabled_cache = True
    elif os.environ.get('GSUTIL_TEST_ANALYTICS') == '2':
        cls._disabled_cache = False
        cls.StartTestCollector()
    elif boto.config.getbool('GSUtil', 'use_gcloud_storage', False):
        cls._disabled_cache = True
    elif system_util.InvokedViaCloudSdk():
        cls._disabled_cache = not os.environ.get('GA_CID')
    elif os.path.exists(_UUID_FILE_PATH):
        with open(_UUID_FILE_PATH) as f:
            cls._disabled_cache = f.read() == _DISABLED_TEXT
    else:
        cls._disabled_cache = True