import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
def assertTranslationEqual(self, expected_tuple, error):
    self.assertEqual(expected_tuple, request._translate_error(error))