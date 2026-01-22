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
def _do_deprecated_test(self, opt_class, value, result, key, section='DEFAULT', dname=None, dgroup=None):
    self.conf.register_opt(opt_class('newfoo', deprecated_name=dname, deprecated_group=dgroup))
    paths = self.create_tempfiles([('test', '[' + section + ']\n' + key + ' = ' + value + '\n')])
    self.conf(['--config-file', paths[0]])
    self.assertTrue(hasattr(self.conf, 'newfoo'))
    self.assertEqual(result, self.conf.newfoo)