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
def test_regex_good(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo', regex='foo|bar'))
    self.conf(['--foo', 'bar'])
    self.assertEqual('bar', self.conf.foo)
    self.conf(['--foo', 'foo'])
    self.assertEqual('foo', self.conf.foo)
    self.conf(['--foo', 'foobar'])
    self.assertEqual('foobar', self.conf.foo)