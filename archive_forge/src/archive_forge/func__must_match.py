import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def _must_match(regex, string, pos):
    match = regex.match(string, pos)
    assert match is not None
    return match