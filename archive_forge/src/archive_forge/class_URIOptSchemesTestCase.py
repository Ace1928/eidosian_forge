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
class URIOptSchemesTestCase(BaseTestCase):

    def test_uriopt_schemes_good(self):
        self.conf.register_cli_opt(cfg.URIOpt('foo', schemes=['http', 'ftp']))
        self.conf(['--foo', 'http://www.example.com'])
        self.assertEqual('http://www.example.com', self.conf.foo)
        self.conf(['--foo', 'ftp://example.com/archives'])
        self.assertEqual('ftp://example.com/archives', self.conf.foo)

    def test_uriopt_schemes_bad(self):
        self.conf.register_cli_opt(cfg.URIOpt('foo', schemes=['http', 'ftp']))
        self.assertRaises(SystemExit, self.conf, ['--foo', 'https://www.example.com'])
        self.assertRaises(SystemExit, self.conf, ['--foo', 'file://www.example.com'])