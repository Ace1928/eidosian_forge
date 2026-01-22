from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import os
import subprocess
from unittest import mock
import six
from gslib import context_config
from gslib import exception
from gslib.tests import testcase
from gslib.tests.testcase import base
from gslib.tests.util import SetBotoConfigForTest
@testcase.integration_testcase.SkipForS3('mTLS only runs on GCS JSON API.')
@testcase.integration_testcase.SkipForXML('mTLS only runs on GCS JSON API.')
class TestPemFileParser(testcase.GsUtilUnitTestCase):
    """Test PEM-format certificate parsing for mTLS."""

    def test_pem_file_with_comment_at_beginning(self):
        sections = context_config._split_pem_into_sections(CERT_KEY_WITH_COMMENT_AT_BEGIN, self.logger)
        self.assertEqual(sections['CERTIFICATE'], CERT_SECTION)
        self.assertEqual(sections['ENCRYPTED PRIVATE KEY'], ENCRYPTED_KEY_SECTION)

    def test_pem_file_with_comment_at_end(self):
        sections = context_config._split_pem_into_sections(CERT_KEY_WITH_COMMENT_AT_END, self.logger)
        self.assertEqual(sections['CERTIFICATE'], CERT_SECTION)
        self.assertEqual(sections['ENCRYPTED PRIVATE KEY'], ENCRYPTED_KEY_SECTION)

    def test_pem_file_with_comment_in_between(self):
        sections = context_config._split_pem_into_sections(CERT_KEY_WITH_COMMENT_IN_BETWEEN, self.logger)
        self.assertEqual(sections['CERTIFICATE'], CERT_SECTION)
        self.assertEqual(sections['ENCRYPTED PRIVATE KEY'], ENCRYPTED_KEY_SECTION)

    def test_pem_file_with_bad_format_embedded_section(self):
        sections = context_config._split_pem_into_sections(BAD_CERT_KEY_EMBEDDED_SECTION, self.logger)
        self.assertIsNone(sections.get('CERTIFICATE'))
        self.assertEqual(sections.get('ENCRYPTED PRIVATE KEY'), ENCRYPTED_KEY_SECTION)

    def test_pem_file_with_bad_format_missing_ending(self):
        sections = context_config._split_pem_into_sections(BAD_CERT_KEY_MISSING_END, self.logger)
        self.assertEqual(sections.get('CERTIFICATE'), CERT_SECTION)
        self.assertIsNone(sections.get('ENCRYPTED PRIVATE KEY'))

    def test_pem_file_with_bad_format_missing_beginning(self):
        sections = context_config._split_pem_into_sections(BAD_CERT_KEY_MISSING_BEGIN, self.logger)
        self.assertIsNone(sections.get('CERTIFICATE'))
        self.assertEqual(sections.get('ENCRYPTED PRIVATE KEY'), ENCRYPTED_KEY_SECTION)

    def test_pem_file_with_bad_format_section_mismatch(self):
        sections = context_config._split_pem_into_sections(BAD_CERT_KEY_MISMATCH, self.logger)
        self.assertIsNone(sections.get('CERTIFICATE'))
        self.assertIsNone(sections.get('ENCRYPTED PRIVATE KEY'))