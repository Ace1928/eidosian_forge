import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
class Test_CICPCanonicalRelpath(tests.TestCaseWithTransport):

    def assertRelpath(self, expected, base, path):
        actual = osutils._cicp_canonical_relpath(base, path)
        self.assertEqual(expected, actual)

    def test_simple(self):
        self.build_tree(['MixedCaseName'])
        base = osutils.realpath(self.get_transport('.').local_abspath('.'))
        self.assertRelpath('MixedCaseName', base, 'mixedcAsename')

    def test_subdir_missing_tail(self):
        self.build_tree(['MixedCaseParent/', 'MixedCaseParent/a_child'])
        base = osutils.realpath(self.get_transport('.').local_abspath('.'))
        self.assertRelpath('MixedCaseParent/a_child', base, 'MixedCaseParent/a_child')
        self.assertRelpath('MixedCaseParent/a_child', base, 'MixedCaseParent/A_Child')
        self.assertRelpath('MixedCaseParent/not_child', base, 'MixedCaseParent/not_child')

    def test_at_root_slash(self):
        if osutils.MIN_ABS_PATHLENGTH > 1:
            raise tests.TestSkipped('relpath requires %d chars' % osutils.MIN_ABS_PATHLENGTH)
        self.assertRelpath('foo', '/', '/foo')

    def test_at_root_drive(self):
        if sys.platform != 'win32':
            raise tests.TestNotApplicable('we can only test drive-letter relative paths on Windows where we have drive letters.')
        self.assertRelpath('foo', 'C:/', 'C:/foo')
        self.assertRelpath('foo', 'X:/', 'X:/foo')
        self.assertRelpath('foo', 'X:/', 'X://foo')