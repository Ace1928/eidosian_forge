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
def to_base64(original):
    """
    Convert a string to base64, via UTF-8. Returns None on invalid input.
    """
    base64_string = None
    try:
        utf8_bytes = original.encode('UTF-8')
        base64_bytes = base64.b64encode(utf8_bytes)
        base64_string = base64_bytes.decode('UTF-8')
    except Exception as err:
        logger.warning('Unable to encode {orig} to base64:'.format(orig=original), err)
    return base64_string