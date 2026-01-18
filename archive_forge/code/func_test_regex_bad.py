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
def test_regex_bad(self):
    self.conf.register_cli_opt(cfg.StrOpt('foo', regex='bar'))
    self.assertRaises(SystemExit, self.conf, ['--foo', 'foo'])