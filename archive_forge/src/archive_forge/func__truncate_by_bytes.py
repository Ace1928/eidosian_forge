import base64
import json
import linecache
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import namedtuple
from copy import copy
from decimal import Decimal
from numbers import Real
from datetime import datetime
from functools import partial
import sentry_sdk
from sentry_sdk._compat import PY2, PY33, PY37, implements_str, text_type, urlparse
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import DEFAULT_MAX_VALUE_LENGTH
def _truncate_by_bytes(string, max_bytes):
    """
    Truncate a UTF-8-encodable string to the last full codepoint so that it fits in max_bytes.
    """
    if isinstance(string, bytes):
        truncated = string[:max_bytes - 3]
    else:
        truncated = string.encode('utf-8')[:max_bytes - 3].decode('utf-8', errors='ignore')
    return truncated + '...'