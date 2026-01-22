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
class UnhandledKeyType:
    """
    A dictionary key of a type that we cannot or do not check for duplicates.
    """