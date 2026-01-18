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
def test_required_cli_opt_with_dash(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo-bar', required=True))
    self.conf(['--foo-bar', 'baz'])
    self.assertTrue(hasattr(self.conf, 'foo_bar'))
    self.assertEqual('baz', self.conf.foo_bar)