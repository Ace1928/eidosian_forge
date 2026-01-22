import copy
import datetime
import sys
import inspect
import warnings
import hashlib
from http.client import HTTPMessage
import logging
import shlex
import re
import os
from collections import OrderedDict
from collections.abc import MutableMapping
from math import floor
from botocore.vendored import six
from botocore.exceptions import MD5UnavailableError
from dateutil.tz import tzlocal
from urllib3 import exceptions
from urllib.parse import (
from http.client import HTTPResponse
from io import IOBase as _IOBase
from base64 import encodebytes
from email.utils import formatdate
from itertools import zip_longest
import json
def _windows_shell_split(s):
    """Splits up a windows command as the built-in command parser would.

    Windows has potentially bizarre rules depending on where you look. When
    spawning a process via the Windows C runtime (which is what python does
    when you call popen) the rules are as follows:

    https://docs.microsoft.com/en-us/cpp/cpp/parsing-cpp-command-line-arguments

    To summarize:

    * Only space and tab are valid delimiters
    * Double quotes are the only valid quotes
    * Backslash is interpreted literally unless it is part of a chain that
      leads up to a double quote. Then the backslashes escape the backslashes,
      and if there is an odd number the final backslash escapes the quote.

    :param s: The command string to split up into parts.
    :return: A list of command components.
    """
    if not s:
        return []
    components = []
    buff = []
    is_quoted = False
    num_backslashes = 0
    for character in s:
        if character == '\\':
            num_backslashes += 1
        elif character == '"':
            if num_backslashes > 0:
                buff.append('\\' * int(floor(num_backslashes / 2)))
                remainder = num_backslashes % 2
                num_backslashes = 0
                if remainder == 1:
                    buff.append('"')
                    continue
            is_quoted = not is_quoted
            buff.append('')
        elif character in [' ', '\t'] and (not is_quoted):
            if num_backslashes > 0:
                buff.append('\\' * num_backslashes)
                num_backslashes = 0
            if buff:
                components.append(''.join(buff))
                buff = []
        else:
            if num_backslashes > 0:
                buff.append('\\' * num_backslashes)
                num_backslashes = 0
            buff.append(character)
    if is_quoted:
        raise ValueError(f'No closing quotation in string: {s}')
    if num_backslashes > 0:
        buff.append('\\' * num_backslashes)
    if buff:
        components.append(''.join(buff))
    return components