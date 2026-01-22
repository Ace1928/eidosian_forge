import datetime
import numbers
import re
import sys
import os
import textwrap
from tornado.escape import _unicode, native_str
from tornado.log import define_logging_options
from tornado.util import basestring_type, exec_in
from typing import (
def add_parse_callback(self, callback: Callable[[], None]) -> None:
    """Adds a parse callback, to be invoked when option parsing is done."""
    self._parse_callbacks.append(callback)