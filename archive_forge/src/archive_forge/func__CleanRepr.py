from the command line:
import functools
import re
import types
import unittest
import uuid
def _CleanRepr(obj):
    return ADDR_RE.sub('<\\1>', repr(obj))