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
def test_sub_command(self):

    def add_parsers(subparsers):
        sub = subparsers.add_parser('a')
        sub.add_argument('bar', type=int)
    self.conf.register_cli_opt(cfg.SubCommandOpt('cmd', handler=add_parsers))
    self.assertTrue(hasattr(self.conf, 'cmd'))
    self.conf(['a', '10'])
    self.assertTrue(hasattr(self.conf.cmd, 'name'))
    self.assertTrue(hasattr(self.conf.cmd, 'bar'))
    self.assertEqual('a', self.conf.cmd.name)
    self.assertEqual(10, self.conf.cmd.bar)