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
def test_positional_opts_order(self):
    self.conf.register_cli_opts((cfg.StrOpt('command', positional=True), cfg.StrOpt('arg1', positional=True), cfg.StrOpt('arg2', positional=True)))
    self.conf(['command', 'arg1', 'arg2'])
    self.assertEqual('command', self.conf.command)
    self.assertEqual('arg1', self.conf.arg1)
    self.assertEqual('arg2', self.conf.arg2)