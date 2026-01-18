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
def test_requires(self):
    attrs = {'name': 'package', 'version': '1.0', 'requires': ['other', 'another (==1.0)']}
    dist = Distribution(attrs)
    self.assertEqual(dist.metadata.get_requires(), ['other', 'another (==1.0)'])
    self.assertEqual(dist.get_requires(), ['other', 'another (==1.0)'])
    meta = self.format_metadata(dist)
    self.assertIn('Metadata-Version: 1.1', meta)
    self.assertNotIn('provides:', meta.lower())
    self.assertIn('Requires: other', meta)
    self.assertIn('Requires: another (==1.0)', meta)
    self.assertNotIn('obsoletes:', meta.lower())