from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
from contextlib import contextmanager
import os
import re
import subprocess
from unittest import mock
from boto import config
from gslib import command
from gslib import command_argument
from gslib import exception
from gslib.commands import rsync
from gslib.commands import version
from gslib.commands import test
from gslib.cs_api_map import ApiSelector
from gslib.tests import testcase
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import shim_util
from gslib.utils import system_util
from gslib.tests import util
class TestGetFlagFromHeader(testcase.GsUtilUnitTestCase):
    """Test Command.get_flag_from_header function.

  We only test the unset functionality because rest of the workflows have been
  already tested indirectly in TestHeaderTranslation.
  """

    def test_get_flag_from_header_with_unset_true_for_data_transfer_headers(self):
        headers_to_expected_flag_map = {'Cache-Control': '--clear-cache-control', 'Content-Disposition': '--clear-content-disposition', 'Content-Encoding': '--clear-content-encoding', 'Content-Language': '--clear-content-language', 'Content-Type': '--clear-content-type', 'custom-time': '--clear-custom-time'}
        for header, expected_flag in headers_to_expected_flag_map.items():
            result = shim_util.get_flag_from_header(header, 'fake_val', unset=True)
            self.assertEqual(result, expected_flag)

    def test_get_flag_from_header_with_unset_true_for_precondition_headers(self):
        """Should return None."""
        for header in ['x-goog-if-generation-match', 'x-goog-if-metageneration-match']:
            result = shim_util.get_flag_from_header(header, 'fake_val', unset=True)
            self.assertIsNone(result)

    def test_get_flag_from_header_with_unset_true_for_content_md5(self):
        """Should return None."""
        result = shim_util.get_flag_from_header('Content-MD5', 'fake_val', unset=True)
        self.assertIsNone(result)

    def test_get_flag_from_header_with_unset_true_for_invalid_header(self):
        """Should return None."""
        result = shim_util.get_flag_from_header('invalid_header', 'fake_val', unset=True)
        self.assertIsNone(result)

    def test_get_flag_from_header_with_unset_true_for_metadata_headers(self):
        """Should return --remove-custom-metadata flag."""
        headers_to_expected_flag_map = {'x-goog-meta-foo': '--remove-custom-metadata=foo', 'x-amz-meta-foo': '--remove-custom-metadata=foo'}
        for header, expected_flag in headers_to_expected_flag_map.items():
            result = shim_util.get_flag_from_header(header, 'fake_val', unset=True)
            self.assertEqual(result, expected_flag)