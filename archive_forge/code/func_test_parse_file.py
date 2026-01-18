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
def test_parse_file(self):
    paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar\n[BLAA]\nbar = foo\n')])
    sections = {}
    parser = cfg.ConfigParser(paths[0], sections)
    parser.parse()
    self.assertIn('DEFAULT', sections)
    self.assertIn('BLAA', sections)
    self.assertEqual(sections['DEFAULT']['foo'], ['bar'])
    self.assertEqual(sections['BLAA']['bar'], ['foo'])