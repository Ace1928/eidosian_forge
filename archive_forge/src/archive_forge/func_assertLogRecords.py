import contextlib
import difflib
import pprint
import pickle
import re
import sys
import logging
import warnings
import weakref
import inspect
import types
from copy import deepcopy
from test import support
import unittest
from unittest.test.support import (
from test.support import captured_stderr, gc_collect
def assertLogRecords(self, records, matches):
    self.assertEqual(len(records), len(matches))
    for rec, match in zip(records, matches):
        self.assertIsInstance(rec, logging.LogRecord)
        for k, v in match.items():
            self.assertEqual(getattr(rec, k), v)