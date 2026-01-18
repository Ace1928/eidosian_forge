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
def test_sub_command_with_group(self):

    def add_parsers(subparsers):
        sub = subparsers.add_parser('a')
        sub.add_argument('--bar', choices='XYZ')
    self.conf.register_cli_opt(cfg.SubCommandOpt('cmd', handler=add_parsers), group='blaa')
    self.assertTrue(hasattr(self.conf, 'blaa'))
    self.assertTrue(hasattr(self.conf.blaa, 'cmd'))
    self.conf(['a', '--bar', 'Z'])
    self.assertTrue(hasattr(self.conf.blaa.cmd, 'name'))
    self.assertTrue(hasattr(self.conf.blaa.cmd, 'bar'))
    self.assertEqual('a', self.conf.blaa.cmd.name)
    self.assertEqual('Z', self.conf.blaa.cmd.bar)