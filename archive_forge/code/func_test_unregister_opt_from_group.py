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
def test_unregister_opt_from_group(self):
    opt = cfg.StrOpt('foo')
    self.conf.register_opt(opt, group='blaa')
    self.assertTrue(hasattr(self.conf, 'blaa'))
    self.assertTrue(hasattr(self.conf.blaa, 'foo'))
    self.conf.unregister_opt(opt, group='blaa')
    self.assertFalse(hasattr(self.conf.blaa, 'foo'))