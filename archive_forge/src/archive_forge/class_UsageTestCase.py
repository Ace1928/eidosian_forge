import argparse
import errno
import functools
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import fixtures
from oslotest import base
import testscenarios
from oslo_config import cfg
from oslo_config import types
class UsageTestCase(BaseTestCase):

    def test_print_usage(self):
        f = io.StringIO()
        self.conf([])
        self.conf.print_usage(file=f)
        self.assertIn('usage: test [-h] [--config-dir DIR] [--config-file PATH] [--version]', f.getvalue())
        self.assertNotIn('somedesc', f.getvalue())
        self.assertNotIn('tepilog', f.getvalue())
        self.assertNotIn('optional:', f.getvalue())

    def test_print_custom_usage(self):
        conf = self.TestConfigOpts()
        self.tempdirs = []
        f = io.StringIO()
        conf([], usage='%(prog)s FOO BAR')
        conf.print_usage(file=f)
        self.assertIn('usage: test FOO BAR', f.getvalue())
        self.assertNotIn('somedesc', f.getvalue())
        self.assertNotIn('tepilog', f.getvalue())
        self.assertNotIn('optional:', f.getvalue())

    def test_print_help(self):
        f = io.StringIO()
        self.conf([])
        self.conf.print_help(file=f)
        self.assertIn('usage: test [-h] [--config-dir DIR] [--config-file PATH] [--version]', f.getvalue())
        self.assertIn('somedesc', f.getvalue())
        self.assertIn('tepilog', f.getvalue())
        self.assertNotIn('optional:', f.getvalue())