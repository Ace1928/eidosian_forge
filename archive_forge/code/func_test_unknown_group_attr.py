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
def test_unknown_group_attr(self):
    self.conf.register_group(cfg.OptGroup('blaa'))
    self.conf([])
    self.assertTrue(hasattr(self.conf, 'blaa'))
    self.assertFalse(hasattr(self.conf.blaa, 'foo'))
    self.assertRaises(cfg.NoSuchOptError, getattr, self.conf.blaa, 'foo')