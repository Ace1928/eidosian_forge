from __future__ import with_statement
import sys
import unittest
from unittest import TestCase
import simplejson
from simplejson import encoder, decoder, scanner
from simplejson.compat import PY3, long_type, b
class BadBool:

    def __bool__(self):
        1 / 0
    __nonzero__ = __bool__