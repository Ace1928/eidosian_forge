import operator
import sys
import types
import unittest
import abc
import pytest
import six
class FlushableStringIO(six.StringIO):

    def __init__(self):
        six.StringIO.__init__(self)
        self.flushed = False

    def flush(self):
        self.flushed = True