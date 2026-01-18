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
def test_required_positional_opt_defined(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo', required=True, positional=True))
    self.useFixture(fixtures.MonkeyPatch('sys.stdout', io.StringIO()))
    self.assertRaises(SystemExit, self.conf, ['--help'])
    self.assertIn(' foo\n', sys.stdout.getvalue())
    self.conf(['bar'])
    self.assertTrue(hasattr(self.conf, 'foo'))
    self.assertEqual('bar', self.conf.foo)