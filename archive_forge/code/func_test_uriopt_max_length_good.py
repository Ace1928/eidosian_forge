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
def test_uriopt_max_length_good(self):
    self.conf.register_cli_opt(cfg.URIOpt('foo', max_length=30))
    self.conf(['--foo', 'http://www.example.com'])
    self.assertEqual('http://www.example.com', self.conf.foo)