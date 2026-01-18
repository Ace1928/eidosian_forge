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
def test_optional_positional_hyphenated_opt_undefined(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo-bar', required=False, positional=True))
    self.useFixture(fixtures.MonkeyPatch('sys.stdout', io.StringIO()))
    self.assertRaises(SystemExit, self.conf, ['--help'])
    self.assertIn(' [foo_bar]\n', sys.stdout.getvalue())
    self.conf([])
    self.assertTrue(hasattr(self.conf, 'foo_bar'))
    self.assertIsNone(self.conf.foo_bar)