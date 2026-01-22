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
@staticmethod
def _GetCID():
    """Gets the client id from the UUID file or the SDK opt-in, or returns None.

    Returns:
      str, The hex string of the client id.
    """
    if os.path.exists(_UUID_FILE_PATH):
        with open(_UUID_FILE_PATH) as f:
            cid = f.read()
        if cid:
            return cid
    return os.environ.get('GA_CID')