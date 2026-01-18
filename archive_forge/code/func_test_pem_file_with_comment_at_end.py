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
def test_pem_file_with_comment_at_end(self):
    sections = context_config._split_pem_into_sections(CERT_KEY_WITH_COMMENT_AT_END, self.logger)
    self.assertEqual(sections['CERTIFICATE'], CERT_SECTION)
    self.assertEqual(sections['ENCRYPTED PRIVATE KEY'], ENCRYPTED_KEY_SECTION)