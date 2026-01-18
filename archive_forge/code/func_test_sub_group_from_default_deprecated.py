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
def test_sub_group_from_default_deprecated(self):
    self.conf.register_group(cfg.OptGroup('ba'))
    self.conf.register_cli_opt(cfg.StrOpt('foo', default='123', deprecated_group='DEFAULT'), group='ba')
    self.conf.register_cli_opt(cfg.IntOpt('r', default='$foo'), group='ba')
    self.conf([])
    self.assertTrue(hasattr(self.conf, 'ba'))
    self.assertTrue(hasattr(self.conf.ba, 'foo'))
    self.assertEqual('123', self.conf.ba.foo)
    self.assertTrue(hasattr(self.conf.ba, 'r'))
    self.assertEqual(123, self.conf.ba.r)