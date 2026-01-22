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
def _getNumberToken(commandStr):
    """Gets the number token at the start of commandStr.

    Given '5hello' returns '5'
    Given '  5hello' returns '  5'
    Given '-42hello' returns '-42'
    Given '+42hello' returns '+42'
    Given '3.14hello' returns '3.14'

    Raises an exception if it can't tokenize a number.
    """
    pattern = re.compile('^(\\s*(\\+|\\-)?\\d+(\\.\\d+)?)')
    mo = pattern.search(commandStr)
    if mo is None:
        raise PyAutoGUIException('Invalid command at index 0: a number was expected')
    return mo.group(1)