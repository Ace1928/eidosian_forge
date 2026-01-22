import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
class RemotedTestCase(unittest.TestCase):
    """A class to represent test cases run in child processes.

    Instances of this class are used to provide the Python test API a TestCase
    that can be printed to the screen, introspected for metadata and so on.
    However, as they are a simply a memoisation of a test that was actually
    run in the past by a separate process, they cannot perform any interactive
    actions.
    """

    def __eq__(self, other):
        try:
            return self.__description == other.__description
        except AttributeError:
            return False

    def __init__(self, description):
        """Create a psuedo test case with description description."""
        self.__description = description

    def error(self, label):
        raise NotImplementedError('%s on RemotedTestCases is not permitted.' % label)

    def setUp(self):
        self.error('setUp')

    def tearDown(self):
        self.error('tearDown')

    def shortDescription(self):
        return self.__description

    def id(self):
        return '%s' % (self.__description,)

    def __str__(self):
        return '%s (%s)' % (self.__description, self._strclass())

    def __repr__(self):
        return "<%s description='%s'>" % (self._strclass(), self.__description)

    def run(self, result=None):
        if result is None:
            result = self.defaultTestResult()
        result.startTest(self)
        result.addError(self, RemoteError(_u('Cannot run RemotedTestCases.\n')))
        result.stopTest(self)

    def _strclass(self):
        cls = self.__class__
        return '%s.%s' % (cls.__module__, cls.__name__)