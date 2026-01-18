import os
import io
import sys
import unittest
import warnings
import textwrap
from unittest import mock
from distutils.dist import Distribution, fix_help_options
from distutils.cmd import Command
from test.support import (
from test.support.os_helper import TESTFN
from distutils.tests import support
from distutils import log
def test_read_metadata(self):
    attrs = {'name': 'package', 'version': '1.0', 'long_description': 'desc', 'description': 'xxx', 'download_url': 'http://example.com', 'keywords': ['one', 'two'], 'requires': ['foo']}
    dist = Distribution(attrs)
    metadata = dist.metadata
    PKG_INFO = io.StringIO()
    metadata.write_pkg_file(PKG_INFO)
    PKG_INFO.seek(0)
    metadata.read_pkg_file(PKG_INFO)
    self.assertEqual(metadata.name, 'package')
    self.assertEqual(metadata.version, '1.0')
    self.assertEqual(metadata.description, 'xxx')
    self.assertEqual(metadata.download_url, 'http://example.com')
    self.assertEqual(metadata.keywords, ['one', 'two'])
    self.assertEqual(metadata.platforms, ['UNKNOWN'])
    self.assertEqual(metadata.obsoletes, None)
    self.assertEqual(metadata.requires, ['foo'])