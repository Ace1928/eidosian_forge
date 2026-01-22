import collections
import functools
import itertools as _itertools
import re
import sys
from builtins import open as _builtin_open
from codecs import BOM_UTF8, lookup
from io import TextIOWrapper
from ._token import (
Tokenize a source reading Python code as unicode strings.

    This has the same API as tokenize(), except that it expects the *readline*
    callable to return str objects instead of bytes.
    