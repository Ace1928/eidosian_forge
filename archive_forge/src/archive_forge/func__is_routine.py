import __future__
import difflib
import inspect
import linecache
import os
import pdb
import re
import sys
import traceback
import unittest
from io import StringIO, IncrementalNewlineDecoder
from collections import namedtuple
def _is_routine(self, obj):
    """
        Safely unwrap objects and determine if they are functions.
        """
    maybe_routine = obj
    try:
        maybe_routine = inspect.unwrap(maybe_routine)
    except ValueError:
        pass
    return inspect.isroutine(maybe_routine)