from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
def _getCommaToken(commandStr):
    """Gets the comma token at the start of commandStr.

    Given ',' returns ','
    Given '  ,', returns '  ,'

    Raises an exception if a comma isn't found.
    """
    pattern = re.compile('^((\\s*),)')
    mo = pattern.search(commandStr)
    if mo is None:
        raise PyAutoGUIException('Invalid command at index 0: a comma was expected')
    return mo.group(1)